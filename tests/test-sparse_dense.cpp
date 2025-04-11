#include <time.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

#include "ggml-alloc.h"
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
int kBatchSize = 32;
int kHidden = 8192;
int kNeuron = 28672;
bool use_gpu = false;
bool squash = false;
int factor = 2;       // 稀疏化为 1 - 1/facotr 

constexpr int kLayer = 10;

float *k_gpu_idx = NULL;

enum ggml_type data_type = GGML_TYPE_Q4_0;

struct test_model {
    struct ggml_tensor *up_w[kLayer];   // 8192 * 28672
    struct ggml_tensor *down_w[kLayer]; // 28672 * 8192
    struct ggml_tensor *squash_up;
    struct ggml_tensor *squash_down;

    void *weight_data = NULL;
    ggml_tallocr_t weight_talloc;

    void *buf_data = NULL;
    ggml_allocr_t gf_alloc;

    struct ggml_context *ctx;
};

void load_model(test_model &model) {
    size_t buffer_size = 0;
    {
        buffer_size += (kHidden * kNeuron) * ggml_type_size(data_type) / ggml_blck_size(data_type) * (2 * kLayer + 2) ;  // tensor weight
        buffer_size += 1024;                                                        // overhead
    }

    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int)sizeof(ggml_tensor));
    printf("%s: weight buffer size = %ld bytes\n", __func__, buffer_size);
    printf("%s: weight buffer size = %ld MB\n", __func__, buffer_size/1024/1024);

    // create a allocator
    model.weight_data = malloc(buffer_size);
    model.weight_talloc = ggml_tallocr_new(model.weight_data, buffer_size, 32);

    struct ggml_init_params params {
        /*.mem_size   =*/ggml_tensor_overhead() * 1024,
            /*.mem_buffer =*/NULL,
            /*.no_alloc   =*/true,
    };

    // create context
    model.ctx = ggml_init(params);

    // create tensors
    for (int i = 0; i < kLayer; i++) {
        model.up_w[i] = ggml_new_tensor_2d(model.ctx, data_type, kHidden, kNeuron);
        model.down_w[i] = ggml_new_tensor_2d(model.ctx, data_type, kHidden, kNeuron);
    }
    model.squash_up = ggml_new_tensor_2d(model.ctx, data_type, kHidden, kNeuron);
    model.squash_down = ggml_new_tensor_2d(model.ctx, data_type, kHidden, kNeuron);

    // if (use_gpu) {
    //     for (int i = 0; i < kLayer; i++) {
    //         model.up_w[i]->backend = GGML_BACKEND_GPU;
    //         model.down_w[i]->backend = GGML_BACKEND_GPU;
    //     }
    // }

    // alloc memory & init weights
    std::vector<std::thread> ths;
    constexpr int init_worker = 32;
    for (int i = 0; i < kLayer; i++) {
        ggml_tallocr_alloc(model.weight_talloc, model.up_w[i]);
        ggml_tallocr_alloc(model.weight_talloc, model.down_w[i]);
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

    // if (use_gpu) {
    //     for (int i = 0; i < kLayer; i++) {
    //         ggml_cuda_transform_tensor(model.up_w[i]->data, model.up_w[i]);
    //         ggml_cuda_transform_tensor(model.down_w[i]->data, model.down_w[i]);
    //     }
    // }
}

struct ggml_cgraph *build_graph(const test_model &model, float *inp, float *pred_inp, int batch_size) {
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

    struct ggml_tensor *x =  ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, kHidden, batch_size);
    ggml_allocr_alloc(model.gf_alloc, x);
    if (inp != NULL) {
      memcpy(x->data, inp, ggml_nbytes(x));
    }
    struct ggml_tensor *pred = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, kNeuron, batch_size);
    ggml_allocr_alloc(model.gf_alloc, pred);
    if (pred_inp != NULL) {
      memcpy(pred->data, pred_inp, ggml_nbytes(pred));
    }

    struct ggml_tensor *gpu_idx = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, kNeuron, batch_size);
    ggml_allocr_alloc(model.gf_alloc, gpu_idx);
    if (pred_inp != NULL) {
      memcpy(gpu_idx->data, k_gpu_idx, ggml_nbytes(gpu_idx));
    }

    struct ggml_tensor *result_up = NULL;
    struct ggml_tensor *result_down = x;
      
    // if (use_gpu) {
    //     X->backend = GGML_BACKEND_GPU;
    // }
    if (!squash) {
      for (int i = 0; i < kLayer; i++) {
          result_up = ggml_mul_mat_idx(ctx0, model.up_w[i], result_down, pred, gpu_idx);

        //   struct ggml_tensor *down_t = ggml_transpose(ctx0, model.down_w[i]);

        //   result_down = ggml_axpy(ctx0, ggml_cont(ctx0, down_t), result_up, pred, gpu_idx);
          result_down = ggml_axpy(ctx0, model.down_w[i], result_up, pred, gpu_idx);
          // if (use_gpu) {
          //     result_up->backend = GGML_BACKEND_GPU;
          //     result_down->backend = GGML_BACKEND_GPU;
          // }
      }
    } else {
      // TODO: squash mul mat
      throw std::runtime_error("not implement");
      for (int i = 0; i < kLayer; i++) {
          result_up = ggml_mul_mat_idx(ctx0, model.up_w[i], result_down, pred, NULL);
          result_down = ggml_axpy(ctx0, model.down_w[i], result_up, pred, NULL);
          // if (use_gpu) {
          //     result_up->backend = GGML_BACKEND_GPU;
          //     result_down->backend = GGML_BACKEND_GPU;
          // }
      }
    }
    // z = (zT)T
    ggml_build_forward_expand(gf, result_down);

    // delete the temporally context used to build the graph
    ggml_free(ctx0);
    return gf;
}

struct ggml_tensor *compute(const test_model &model, ggml_allocr_t allocr, float *inp, float *pred, int batch_size) {
    struct ggml_cgraph *gf = build_graph(model, inp, pred, batch_size);
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
    if (argc < 4) {
        fprintf(stderr, "Usage: <cmd> <squash> <batch_size> <factor>");
        return -1;
    }

    squash = atoi(argv[1]) == 1;
    kBatchSize = atoi(argv[2]);
    factor = atoi(argv[3]);

    constexpr int init_worker = 64;
    std::vector<std::thread> ths;
    // x
    float *matrix_X = (float *)malloc(sizeof(float) * kBatchSize * kHidden);
    ths.reserve(init_worker);
    for (int i = 0; i < init_worker; i++) {
        ths.emplace_back([matrix_X, i]() {
            int per = kBatchSize * kHidden / init_worker;
            for (int j = 0; j < per; j++) {
                matrix_X[j + i * per] = frand();
            }
        });
    }
    for (auto &th : ths) {
        th.join();
    }

    // prediactor
    float *pred = (float *)malloc(sizeof(float) * kBatchSize * kNeuron);
    for (int i = 0; i < kBatchSize; i++) {
      for (int j = 0; j < kNeuron; j++) {
        pred[i * kNeuron + j] = j % factor ? -1 : 1;
      }
    }

    // gpu_idx
    k_gpu_idx = (float *)malloc(sizeof(float) * kBatchSize * kNeuron);
    for (int i = 0; i < kBatchSize * kNeuron; i++) {
        k_gpu_idx[i] = 0.0;
    }
    test_model model;
    load_model(model);
    stat_init();
    model.gf_alloc = ggml_allocr_new_measure(32);

    {
        // create the worst case graph for memory usage estimation
        struct ggml_cgraph *gf = build_graph(model, NULL, NULL, kBatchSize);

        // compute the required memory
        size_t alloc_size = ggml_allocr_alloc_graph(model.gf_alloc, gf) + 32;
        fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__, alloc_size / 1024.0f / 1024.0f);
        fprintf(stderr, "%s: compute buffer size: %.2f B\n", __func__, alloc_size / 1.0f);
        model.buf_data = malloc(alloc_size);
        ggml_allocr_free(model.gf_alloc);
        model.gf_alloc = ggml_allocr_new(model.buf_data, alloc_size, 32);
        ggml_cuda_set_scratch_size(alloc_size);
    }
    {
        // warm up
        ggml_allocr_reset(model.gf_alloc);
        struct ggml_tensor *result = compute(model, model.gf_alloc, matrix_X, pred, kBatchSize);
        (void)result;
    }
    llama_stat_reset();
    printf("\nStart Compute...\n");
    printf("Matrix X: [%i, %i]\n", kBatchSize, kHidden);
    printf("Matrix Weight: [%i, %i]\n", kHidden, kNeuron);

    const int64_t start_us = ggml_time_us();
    {
        ggml_allocr_reset(model.gf_alloc);
        struct ggml_tensor *result = compute(model, model.gf_alloc, matrix_X, pred, kBatchSize);
        (void)result;
    }
    llama_show_stat();
    const int64_t end_us = ggml_time_us();
    printf("%s: elapsed us:    %d / %f ms\n", __func__, (int)(end_us - start_us), (end_us - start_us) / 1000.0);

    // free memory
    ggml_cuda_free_scratch();
    ggml_free(model.ctx);
    ggml_allocr_free(model.gf_alloc);
    ggml_tallocr_free(model.weight_talloc);
    free(matrix_X);
    free(pred);
    return 0;
}