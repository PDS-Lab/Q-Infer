#include <cstdint>
#include <cstdlib>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "statistic.h"

// #define GGML_USE_CUBLAS // uncomment this to use cuda backend, make sure
// build ggml lib with GGML_CUBLAS=ON

#ifdef GGML_USE_CUBLAS
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

float frand(void) { return (float)rand() / (float)RAND_MAX; }

// [M, N] * [N, K];
int M = 32;
int N = 8192;
int K = 8192;
bool use_gpu = false;

constexpr int kLayer = 4;

struct test_model {
    struct ggml_tensor *up_w[kLayer];
    struct ggml_tensor *down_w[kLayer];
    void *weight_data = NULL;
    void *buf_data = NULL;
    ggml_tallocr_t talloc;
    struct ggml_context *ctx;
};

void load_model(test_model &model) {
    size_t buffer_size = 0;
    {
        buffer_size += 2 * (N * K) * ggml_type_size(GGML_TYPE_Q4_0) * kLayer / 32;  // tensor weight
        buffer_size += 1024;                                                        // overhead
    }

    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int)sizeof(ggml_tensor));
    printf("%s: weight buffer size = %ld bytes\n", __func__, buffer_size);
    printf("%s: weight buffer size = %ld MB\n", __func__, buffer_size/1024/1024);

    // create a allocator
    model.weight_data = malloc(buffer_size);
    model.talloc = ggml_tallocr_new(model.weight_data, buffer_size, 32);

    struct ggml_init_params params {
        /*.mem_size   =*/ggml_tensor_overhead() * 1024,
            /*.mem_buffer =*/NULL,
            /*.no_alloc   =*/true,
    };

    // create context
    model.ctx = ggml_init(params);

    // create tensors
    for (int i = 0; i < kLayer; i++) {
        model.up_w[i] = ggml_new_tensor_2d(model.ctx, GGML_TYPE_Q4_0, N, K);
        model.down_w[i] = ggml_new_tensor_2d(model.ctx, GGML_TYPE_Q4_0, K, N);
    }

    if (use_gpu) {
        for (int i = 0; i < kLayer; i++) {
            model.up_w[i]->backend = GGML_BACKEND_GPU;
            model.down_w[i]->backend = GGML_BACKEND_GPU;
        }
    }

    // alloc memory & init weights
    std::vector<std::thread> ths;
    constexpr int init_worker = 32;
    for (int i = 0; i < kLayer; i++) {
        ggml_tallocr_alloc(model.talloc, model.up_w[i]);
        ggml_tallocr_alloc(model.talloc, model.down_w[i]);
        assert(ggml_nbytes(model.up_w[i]) % init_worker == 0);
        assert(ggml_nbytes(model.down_w[i]) % init_worker == 0);
        int up_cnt = ggml_nbytes(model.up_w[i]) / init_worker;
        int down_cnt = ggml_nbytes(model.up_w[i]) / init_worker;
        for (int j = 0; j < init_worker; j++) {
            ths.emplace_back([j, &model, up_cnt, i, down_cnt]() {
                int seed = rand();
                for (int c = 0; c < up_cnt; c++) {
                    ((char *)(model.up_w[i]->data))[c + j * up_cnt] = seed + c;
                }
                seed = rand();
                for (int c = 0; c < down_cnt; c++) {
                    ((char *)(model.down_w[i]->data))[c + j * down_cnt] = seed + c;
                }
            });
        }
    }
    for (auto &th : ths) {
        th.join();
    }

    if (use_gpu) {
        for (int i = 0; i < kLayer; i++) {
            ggml_cuda_transform_tensor(model.up_w[i]->data, model.up_w[i]);
            ggml_cuda_transform_tensor(model.down_w[i]->data, model.down_w[i]);
        }
    }
}

struct ggml_cgraph *build_graph(const test_model &model, struct ggml_tensor *X) {
    static size_t buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/buf.data(),
        /*.no_alloc   =*/true,  // the tensors will be allocated later by
                                // ggml_gallocr_alloc_graph()
    };

    // create a temporally context to build the graph
    struct ggml_context *ctx0 = ggml_init(params0);

    struct ggml_cgraph *gf = ggml_new_graph(ctx0);

    // zT = wT @ x
    struct ggml_tensor *result_up = NULL;
    struct ggml_tensor *result_down = X;
    if (use_gpu) {
        X->backend = GGML_BACKEND_GPU;
    }
    for (int i = 0; i < kLayer; i++) {
        result_up = ggml_mul_mat(ctx0, model.up_w[i], result_down);
        result_down = ggml_mul_mat(ctx0, model.down_w[i], result_up);
        if (use_gpu) {
            result_up->backend = GGML_BACKEND_GPU;
            result_down->backend = GGML_BACKEND_GPU;
        }
    }
    // z = (zT)T
    ggml_build_forward_expand(gf, result_down);

    // delete the temporally context used to build the graph
    ggml_free(ctx0);
    return gf;
}

struct ggml_tensor *compute(const test_model &model, ggml_allocr_t allocr, struct ggml_tensor *X) {
    struct ggml_cgraph *gf = build_graph(model, X);
    // allocate tensors
    ggml_allocr_alloc_graph(allocr, gf);
    for (int i = 0; i < gf->n_leafs; i++) {
        ggml_tensor *node = gf->leafs[i];
        if (node->backend == GGML_BACKEND_GPU && node->extra == NULL) {
            ggml_cuda_assign_scratch_offset(node,  (char*)node->data - (char *) model.buf_data);
            ggml_cuda_copy_to_device(node);
        }
    }

    for (int i = 0; i < gf->n_nodes; i++) {
        ggml_tensor *node = gf->nodes[i];
        if (node->backend == GGML_BACKEND_GPU && node->extra == NULL) {
            ggml_cuda_assign_scratch_offset(node,  (char*)node->data - (char *) model.buf_data);
        }
    }

    int n_threads = 1;
    if (!use_gpu) {
        n_threads = 48;
    }

    struct ggml_cplan plan = ggml_graph_plan(gf, n_threads);

    std::vector<uint8_t> worker_buf;
    if (plan.work_size > 0) {
        worker_buf.resize(plan.work_size);
        plan.work_data = worker_buf.data();
    }
    ggml_graph_compute(gf, &plan);
    llama_statistic(gf);

    // ggml_graph_print(gf);

    // in this case, the output tensor is the last one in the graph
    return gf->nodes[gf->n_nodes - 1];
}

int main(int argc, char **argv) {
    ggml_time_init();
    if (argc < 3) {
        fprintf(stderr, "Usage: <cmd> <gpu_flag> <batch_size>");
        return -1;
    }

    use_gpu = atoi(argv[1]) == 1;
    M = atoi(argv[2]);

    constexpr int init_worker = 64;
    std::vector<std::thread> ths;
    // x
    float *matrix_X = (float *)malloc(sizeof(float) * M * N);
    ths.reserve(init_worker);
    for (int i = 0; i < init_worker; i++) {
        ths.emplace_back([matrix_X, i]() {
            int per = M * N / init_worker;
            for (int j = 0; j < per; j++) {
                matrix_X[j + i * per] = frand();
            }
        });
    }
    for (auto &th : ths) {
        th.join();
    }


    test_model model;
    load_model(model);
    stat_init();
    ggml_allocr_t allocr = ggml_allocr_new_measure(32);

    {
        // create the worst case graph for memory usage estimation
        struct ggml_tensor *x = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, N, M);
        ggml_allocr_alloc(allocr, x);
        struct ggml_cgraph *gf = build_graph(model, x);

        // compute the required memory
        size_t alloc_size = ggml_allocr_alloc_graph(allocr, gf) + 32;
        fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__, alloc_size / 1024.0f / 1024.0f);
        fprintf(stderr, "%s: compute buffer size: %.2f B\n", __func__, alloc_size / 1.0f);
        model.buf_data = malloc(alloc_size);
        allocr = ggml_allocr_new(model.buf_data, alloc_size, 32);
        ggml_cuda_set_scratch_size(alloc_size);
    }
    {
        // warm up
        ggml_allocr_reset(allocr);
        struct ggml_tensor *x = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, N, M);
        if (use_gpu) {
            x->backend = GGML_BACKEND_GPU;
        }
        ggml_allocr_alloc(allocr, x);
        memcpy(x->data, matrix_X, ggml_nbytes(x));
        struct ggml_tensor *result = compute(model, allocr, x);
        (void)result;
    }
    llama_stat_reset();
    printf("\nStart Compute...\n");
    printf("Matrix X: [%i, %i]\n", M, N);
    printf("Matrix Weight: [%i, %i]\n", N, K);

    const int64_t start_us = ggml_time_us();
    {
        ggml_allocr_reset(allocr);
        struct ggml_tensor *x = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, N, M);
        if (use_gpu) {
            x->backend = GGML_BACKEND_GPU;
        }
        ggml_allocr_alloc(allocr, x);
        memcpy(x->data, matrix_X, ggml_nbytes(x));
        struct ggml_tensor *result = compute(model, allocr, x);
        (void)result;
    }
    llama_show_stat();
    const int64_t end_us = ggml_time_us();
    printf("%s: elapsed us:    %d / %f ms\n", __func__, (int)(end_us - start_us), (end_us - start_us) / 1000.0);

    // free memory
    ggml_free(model.ctx);
    ggml_allocr_free(allocr);
    free(model.weight_data);
    free(model.buf_data);
    return 0;
}