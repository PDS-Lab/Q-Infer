/*
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "ggml-cann.h"

#include <acl/acl.h>
#include <stdarg.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <climits>

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cann/aclnn_ops.h"
#include "ggml-cann/common.h"


[[noreturn]] void ggml_cann_error(const char* stmt, const char* func,
                                  const char* file, int line, const char* msg) {
    int32_t id = -1;
    aclrtGetDevice(&id);

    printf("CANN error: %s\n", msg);
    printf("  current device: %d, in function %s at %s:%d\n", id, func,
            file, line);
    printf("  %s\n", stmt);
    GGML_ASSERT(false && "CANN error");
}

#define dev_sparse_threshold 0
int g_main_device = 0;
static aclrtStream g_cannStreams[GGML_CANN_MAX_DEVICES][MAX_STREAMS] = { nullptr };
static ggml_backend_cann_context* g_ctx; 
static uint32_t g_device_count = -1;
static int g_compute_capabilities[GGML_CANN_MAX_DEVICES];
static float g_tensor_split[GGML_CANN_MAX_DEVICES] = {0};
static aclrtContext g_cublas_handles[GGML_CANN_MAX_DEVICES] = {nullptr};

static void * g_scratch_buffer = nullptr;
static size_t g_scratch_size = 0;
static size_t g_scratch_offset = 0;

static bool g_cann_loaded = false;

void ggml_cann_set_device(int32_t device) {
    // int current_device;
    // ACL_CHECK(aclrtGetDevice(&current_device));

    // if (device == current_device) {
    //     return;
    // }

    ACL_CHECK(aclrtSetDevice(device));
}

static int64_t get_row_rounding(ggml_type type) {
    int64_t min_compute_capability = INT_MAX;
    int64_t max_compute_capability = INT_MIN;
    for (int64_t id = 0; id < g_device_count; ++id) {
        if (g_tensor_split[id] < (id + 1 < g_device_count ? g_tensor_split[id + 1] : 1.0f)) {
            if (min_compute_capability > g_compute_capabilities[id]) {
                min_compute_capability = g_compute_capabilities[id];
            }
            if (max_compute_capability < g_compute_capabilities[id]) {
                max_compute_capability = g_compute_capabilities[id];
            }
        }
    }
    #define CC_VOLTA      700
    switch(type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
            return max_compute_capability >= CC_VOLTA ? 128 : 64;
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return 64;
        case GGML_TYPE_F16:
            return 1;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
            return max_compute_capability >= CC_VOLTA ? 128 : 64;
        case GGML_TYPE_Q6_K:
            return 64;
        default:
            GGML_ASSERT(false);
    }
}


// extra ctx pool for CANN
#define GGML_CANN_MAX_NODES 8192

// buffer pool for CANN
#define MAX_CANN_BUFFERS 256

struct scoped_spin_lock {
    std::atomic_flag& lock;
    scoped_spin_lock(std::atomic_flag& lock) : lock(lock) {
        while (lock.test_and_set(std::memory_order_acquire)) {
            ; // spin
        }
    }
    ~scoped_spin_lock() {
        lock.clear(std::memory_order_release);
    }
    scoped_spin_lock(const scoped_spin_lock&) = delete;
    scoped_spin_lock& operator=(const scoped_spin_lock&) = delete;
};

struct cann_buffer {
    void * ptr = nullptr;
    size_t size = 0;
};

static cann_buffer g_cann_buffer_pool[GGML_CANN_MAX_DEVICES][MAX_CANN_BUFFERS];
static std::atomic_flag g_cann_pool_lock = ATOMIC_FLAG_INIT;

// todo: wrapped in a ggml_cann_pool
static void * ggml_cann_pool_malloc(size_t size, size_t * actual_size) {
    scoped_spin_lock lock(g_cann_pool_lock);
    int id;
    ACL_CHECK(aclrtGetDevice(&id));

    size_t best_diff = 1ull << 36;
    int ibest = -1;
    for (int i = 0; i < MAX_CANN_BUFFERS; ++i) {
        cann_buffer& b = g_cann_buffer_pool[id][i];
        if (b.ptr != nullptr) {
            if (b.size >= size) {
                size_t diff = b.size - size;
                if (diff < best_diff) {
                    best_diff = diff;
                    ibest = i;
                    if (!best_diff) {
                        void * ptr = b.ptr;
                        *actual_size = b.size;
                        b.ptr = nullptr;
                        b.size = 0;
                        return ptr;
                    }
                }
            }
        }
    }
    if (ibest >= 0) {
        cann_buffer& b = g_cann_buffer_pool[id][ibest];
        void * ptr = b.ptr;
        *actual_size = b.size;
        b.ptr = nullptr;
        b.size = 0;
        return ptr;
    }
    void * ptr;
    size_t look_ahead_size = (size_t) (1.05 * size);
    look_ahead_size = 256 * ((look_ahead_size + 255)/256);
    ACL_CHECK(aclrtMalloc((void **) &ptr, look_ahead_size, ACL_MEM_MALLOC_HUGE_FIRST));
    *actual_size = look_ahead_size;
    return ptr;
}

static void ggml_cann_pool_free(void * ptr, size_t size) {
    scoped_spin_lock lock(g_cann_pool_lock);
    int id;
    ACL_CHECK(aclrtGetDevice(&id));

    for (int i = 0; i < MAX_CANN_BUFFERS; ++i) {
        cann_buffer& b = g_cann_buffer_pool[id][i];
        if (b.ptr == nullptr) {
            b.ptr = ptr;
            b.size = size;
            return;
        }
    }
    fprintf(stderr, "WARNING: cann buffer pool full, increase MAX_CANN_BUFFERS\n");
    ACL_CHECK(aclrtFree(ptr));
}

static void ggml_cann_transform_tensor_impl(void * data, struct ggml_tensor * tensor, bool alloc_only) {
    const int64_t nrows = ggml_nrows(tensor);

    const int64_t ne0 = tensor->ne[0];

    const size_t nb1 = tensor->nb[1];

    ggml_backend_type backend = tensor->backend;
    ggml_tensor_extra_gpu * extra = new struct ggml_tensor_extra_gpu;
    memset(extra, 0, sizeof(*extra));

    for (int64_t id = 0; id < g_device_count; ++id) {
        if (backend == GGML_BACKEND_GPU && id != g_main_device) {
            continue;
        }

        ggml_cann_set_device(id);

        int64_t row_low, row_high;
        if (backend == GGML_BACKEND_GPU) {
            row_low = 0;
            row_high = nrows;
        } else if (backend == GGML_BACKEND_GPU_SPLIT) {
            const int64_t rounding = get_row_rounding(tensor->type);

            row_low = id == 0 ? 0 : nrows*g_tensor_split[id];
            row_low -= row_low % rounding;

            if (id == g_device_count - 1) {
                row_high = nrows;
            } else {
                row_high = nrows*g_tensor_split[id + 1];
                row_high -= row_high % rounding;
            }
        } else {
            GGML_ASSERT(false);
        }
        if (row_low == row_high) {
            continue;
        }

        int64_t nrows_split = row_high - row_low;

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        #define MATRIX_ROW_PADDING 512
        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += (MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING)
                * ggml_type_size(tensor->type)/ggml_blck_size(tensor->type);
        }

        void * buf;
        ACL_CHECK(aclrtMalloc(&buf, size, ACL_MEM_MALLOC_HUGE_FIRST));

        // set padding to 0 to avoid possible NaN values
        if (size > original_size) {
            ACL_CHECK(aclrtMemset(buf + original_size, size - original_size, 0, size - original_size));
        }

        if (!alloc_only) {
            char * buf_host = (char*)data + offset_split;
            ACL_CHECK(aclrtMemcpy(buf, original_size, buf_host, original_size, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        extra->data_device[id] = buf;

        if (backend == GGML_BACKEND_GPU_SPLIT) {
            for (int64_t is = 0; is < MAX_STREAMS; ++is) {
                ACL_CHECK(aclrtCreateEvent(&extra->events[id][is]));
            }
        }
    }

    tensor->extra = extra;
}

static ggml_tensor_extra_gpu * g_temp_tensor_extras = nullptr;
static size_t g_temp_tensor_extra_index = 0;

static ggml_tensor_extra_gpu * ggml_cann_alloc_temp_tensor_extra() {
    if (g_temp_tensor_extras == nullptr) {
        g_temp_tensor_extras = new ggml_tensor_extra_gpu[GGML_CANN_MAX_NODES];
    }

    size_t alloc_index = g_temp_tensor_extra_index;
    g_temp_tensor_extra_index = (g_temp_tensor_extra_index + 1) % GGML_CANN_MAX_NODES;
    ggml_tensor_extra_gpu * extra = &g_temp_tensor_extras[alloc_index];
    memset(extra, 0, sizeof(*extra));

    return extra;
}

static void ggml_cann_assign_buffers_impl(struct ggml_tensor * tensor, bool scratch, bool force_inplace, bool no_alloc) {
    if (scratch && g_scratch_size == 0) {
        return;
    }

    tensor->backend = GGML_BACKEND_GPU;

    // recursively assign cann buffers until a compute tensor is found
    if (tensor->src[0] != nullptr && tensor->src[0]->backend == GGML_BACKEND_CPU) {
        const ggml_op src0_op = tensor->src[0]->op;
        if (src0_op == GGML_OP_RESHAPE || src0_op == GGML_OP_TRANSPOSE || src0_op == GGML_OP_VIEW || src0_op == GGML_OP_PERMUTE) {
            ggml_cann_assign_buffers_impl(tensor->src[0], scratch, force_inplace, no_alloc);
        }
    }
    if (tensor->op == GGML_OP_CPY && tensor->src[1]->backend == GGML_BACKEND_CPU) {
        ggml_cann_assign_buffers_impl(tensor->src[1], scratch, force_inplace, no_alloc);
    }

    if (scratch && no_alloc) {
        return;
    }

    ggml_tensor_extra_gpu * extra;

    const bool inplace = (tensor->src[0] != nullptr && tensor->src[0]->data == tensor->data) ||
        tensor->op == GGML_OP_VIEW ||
        force_inplace;
    const size_t size = ggml_nbytes(tensor);

    ggml_cann_set_device(g_main_device);
    if (inplace && (tensor->src[0]->backend == GGML_BACKEND_GPU || tensor->src[0]->backend == GGML_BACKEND_GPU_SPLIT)) {
        ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu * ) tensor->src[0]->extra;
        char * src0_ddc = (char *) src0_extra->data_device[g_main_device];
        size_t offset = 0;
        if (tensor->op == GGML_OP_VIEW) {
            memcpy(&offset, tensor->op_params, sizeof(size_t));
        }
        extra = ggml_cann_alloc_temp_tensor_extra();
        extra->data_device[g_main_device] = src0_ddc + offset;
    } else if (tensor->op == GGML_OP_CPY) {
        ggml_tensor_extra_gpu * src1_extra = (ggml_tensor_extra_gpu * ) tensor->src[1]->extra;
        void * src1_ddv = src1_extra->data_device[g_main_device];
        extra = ggml_cann_alloc_temp_tensor_extra();
        extra->data_device[g_main_device] = src1_ddv;
    } else if (scratch) {
        GGML_ASSERT(size <= g_scratch_size);
        if (g_scratch_offset + size > g_scratch_size) {
            g_scratch_offset = 0;
        }

        void * data = (void *) g_scratch_buffer;
        if (data == nullptr) {
            ACL_CHECK(aclrtMalloc(&data, g_scratch_size, ACL_MEM_MALLOC_HUGE_FIRST));
            g_scratch_buffer = data;
        }
        extra = ggml_cann_alloc_temp_tensor_extra();
        extra->data_device[g_main_device] = data + g_scratch_offset;

        g_scratch_offset += size;

        GGML_ASSERT(g_scratch_offset <= g_scratch_size);
    } else { // allocate new buffers outside of scratch
        void * data;
        ACL_CHECK(aclrtMalloc(&data, size, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemset(data, size, 0, size));
        extra = new ggml_tensor_extra_gpu;
        memset(extra, 0, sizeof(*extra));
        extra->data_device[g_main_device] = data;
    }

    tensor->extra = extra;
}



void ggml_init_cann(){
    static bool initialized = false;

    ACL_CHECK(aclInit(nullptr));
    if (!initialized) {
        if (aclrtGetDeviceCount(&g_device_count) != ACL_SUCCESS) {
            initialized = true;
            g_cann_loaded = false;
            GGML_ASSERT(false && "Failed to get cann device count");
            return;
        }
        g_device_count = 1;

        GGML_ASSERT(g_device_count <= GGML_CANN_MAX_DEVICES);
        int64_t total_vram = 0;

        fprintf(stderr, "%s: found %d " GGML_CANN_NAME " devices:\n", __func__, g_device_count);
        for (int id = 0; id < g_device_count; ++id) {
            size_t free_mem = ggml_cann_get_free_memory(id);
            // CANNDeviceProp prop;
            // ACL_CHECK(CANNGetDeviceProperties(&prop, id));
            // fprintf(stderr, "  Device %d: %s, compute capability %d.%d\n", id, prop.name, prop.major, prop.minor);

            g_tensor_split[id] = total_vram;
            // total_vram += prop.totalGlobalMem;
            total_vram += free_mem;
            // g_compute_capabilities[id] = 100*prop.major + 10*prop.minor;
            g_compute_capabilities[id] = 100;
        }
        for (int id = 0; id < g_device_count; ++id) {
            g_tensor_split[id] /= total_vram;
        }

        for (int id = 0; id < g_device_count; ++id) {
            ggml_cann_set_device(id);

            // create CANN streams
            for (int is = 0; is < MAX_STREAMS; ++is) {
                ACL_CHECK(aclrtCreateStream(&g_cannStreams[id][is]));
                // ACL_CHECK(CANNStreamCreateWithFlags(&g_cannStreams[id][is], CANNStreamNonBlocking));
            }

            // create cann handle 即 context
            ACL_CHECK(aclrtCreateContext(&g_cublas_handles[id], id));
            // ACL_CHECK(cannCreate(&g_cann_handles[id]));
            // ACL_CHECK(cannSetMathMode(g_cann_handles[id], cann_TF32_TENSOR_OP_MATH));
        }

        // configure logging to stdout
        // ACL_CHECK(cannLoggerConfigure(1, 1, 0, nullptr));

        initialized = true;
        g_cann_loaded = true;
    }
    g_main_device = 0;
    g_ctx = new ggml_backend_cann_context(g_main_device, g_cannStreams[g_main_device]);
}

bool ggml_cann_loaded(void) {
    return g_cann_loaded;
}


void * ggml_cann_host_malloc(size_t size) {
    if (getenv("GGML_CANN_NO_PINNED") != nullptr) {
        return nullptr;
    }

    void * hostPtr = nullptr;
    aclError err = aclrtMallocHost((void **) &hostPtr, size);
    if (err != ACL_SUCCESS) {
        printf("%s: failed to allocate %.2f MiB of pinned memory.\n", __func__,
                           size / 1024.0 / 1024.0);
        return nullptr;
    }
    printf("%s: allocated %.2f MiB of pinned memory at %p.\n", __func__,
                       size / 1024.0 / 1024.0, hostPtr);
    return hostPtr;
}

void ggml_cann_host_free(void * ptr){
    printf("%s: freeing pinned memory at %p.\n", __func__, ptr);
    GGML_ASSERT(ptr != nullptr);
    ACL_CHECK(aclrtFreeHost(ptr));
}

bool   ggml_cann_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst){
    if (!g_cann_loaded) return false;

    const int64_t ne10 = src1->ne[0];
    const int64_t ne01 = src0->ne[1];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    // TODO: find the optimal values for these
    return (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) &&
            src1->type == GGML_TYPE_F32 &&
             dst->type == GGML_TYPE_F32 &&
            (ne0 >= 32 && ne1 >= 32 && ne10 >= 32);
}
void   ggml_cann_set_tensor_split(const float * tensor_split){
    GGML_ASSERT(false && "not implemented");
}


void ggml_cann_transform_tensor(void * data, struct ggml_tensor * tensor) {
    return ggml_cann_transform_tensor_impl(data, tensor, false);
}

void ggml_cann_alloc_tensor(struct ggml_tensor * tensor) {
    return ggml_cann_transform_tensor_impl(nullptr, tensor, true);
}

void ggml_cann_free_data(struct ggml_tensor * tensor) {
    if (!tensor || (tensor->backend != GGML_BACKEND_GPU && tensor->backend != GGML_BACKEND_GPU_SPLIT) ) {
        return;
    }

    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) tensor->extra;

    for (int64_t id = 0; id < g_device_count; ++id) {
        if (extra->data_device[id] != nullptr) {
            ggml_cann_set_device(id);
            ACL_CHECK(aclrtFree(extra->data_device[id]));
        }

        for (int64_t is = 0; is < MAX_STREAMS; ++is) {
            if (extra->events[id][is] != nullptr) {
                ggml_cann_set_device(id);
                ACL_CHECK(aclrtDestroyEvent(extra->events[id][is]));
            }
        }
    }

    delete extra;
}

void   ggml_cann_stream_synchronize(const int stream){
    ggml_cann_set_device(g_main_device);
    ACL_CHECK(aclrtSynchronizeStream(g_cannStreams[g_main_device][stream]));
}


static aclError ggml_cann_cpy_tensor_1d(
    void * dst, const struct ggml_tensor * src, int64_t i1_low, int64_t i1_high, aclrtStream stream) {
    aclrtMemcpyKind kind;
    char * src_ptr;
    if (src->backend == GGML_BACKEND_CPU) {
        kind = ACL_MEMCPY_HOST_TO_DEVICE;
        src_ptr = (char *) src->data;
    } else if (src->backend == GGML_BACKEND_GPU || src->backend == GGML_BACKEND_GPU_SPLIT) {
        GGML_ASSERT(src->backend != GGML_BACKEND_GPU_SPLIT || (i1_low == 0 && i1_high == src->ne[1]));
        kind = ACL_MEMCPY_DEVICE_TO_DEVICE;
        struct ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src->extra;
        int id;
        ACL_CHECK(aclrtGetDevice(&id));
        src_ptr = (char *) extra->data_device[id];
    } else {
        GGML_ASSERT(false);
    }

    char * dst_ptr = (char *) dst;

    const int64_t ne0 = src->ne[0];
    const int64_t nb0 = src->nb[0];
    const int64_t blck = ggml_blck_size(src->type);

    const enum ggml_type type = src->type;
    const int64_t ts = ggml_type_size(type);
    const int64_t bs = ggml_blck_size(type);
    int64_t i1_diff = i1_high - i1_low;

    const char * x = src_ptr + i1_low*nb0/blck;
    return aclrtMemcpyAsync(dst_ptr, i1_diff*nb0/blck, x, i1_diff*nb0/blck, kind, stream);
}

void   ggml_cann_cpy_ptr_1d(struct ggml_tensor *dst, const struct ggml_tensor *src, int host_i, int device_i){
    // ggml_cuda_set_device(g_main_device);
    const aclrtStream load_stream = g_cannStreams[g_main_device][1];
    GGML_ASSERT(src->backend == GGML_BACKEND_CPU && dst->backend == GGML_BACKEND_GPU);

    enum ggml_type type = src->type;
    const int64_t ne0 = src->ne[0];
    const int64_t blck = ggml_blck_size(type);
    const size_t row_data_size = ne0 * ggml_type_size(type) / blck;

    struct ggml_tensor_extra_gpu *dst_extra = (ggml_tensor_extra_gpu *)dst->extra;
    GGML_ASSERT(src->data != NULL && dst_extra->data_device[0] != NULL);

    void *src_ptr = (char *)src->data + row_data_size * host_i;
    void *dst_ptr = (char *)dst_extra->data_device[0] + row_data_size * device_i;

    ACL_CHECK(aclrtMemcpyAsync(dst_ptr, ne0 * ggml_type_size(type) / blck, src_ptr, ne0 * ggml_type_size(type) / blck, ACL_MEMCPY_HOST_TO_DEVICE, load_stream));
}
void   ggml_cann_cpy_1d(struct ggml_tensor *dst, const struct ggml_tensor *src){
    ggml_cann_set_device(g_main_device);
    const aclrtStream main_stream = g_cannStreams[g_main_device][0];

    // TODO: only supports CPU -> GPU as of now
    GGML_ASSERT(src->backend == GGML_BACKEND_CPU && dst->backend == GGML_BACKEND_GPU);
    struct ggml_tensor_extra_gpu * dst_extra = (ggml_tensor_extra_gpu *) dst->extra;

    ACL_CHECK(ggml_cann_cpy_tensor_1d(dst_extra->data_device[0], src, 0, src->ne[0], main_stream));
}
void **ggml_cann_get_data_pp(struct ggml_tensor * tensor){
    // only supports one device for now
    GGML_ASSERT(tensor->backend == GGML_BACKEND_GPU);
    struct ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) tensor->extra;
    return &extra->data_device[0];
}


void ggml_cann_assign_scratch_offset(struct ggml_tensor * tensor, size_t offset) {
    if (g_scratch_size == 0) {
        return;
    }
    if (g_scratch_buffer == nullptr) {
        ggml_cann_set_device(g_main_device);
        ACL_CHECK(aclrtMalloc(&g_scratch_buffer, g_scratch_size, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    ggml_tensor_extra_gpu * extra = ggml_cann_alloc_temp_tensor_extra();

    const bool inplace = (tensor->src[0] != nullptr && tensor->src[0]->data == tensor->data) ||
        tensor->op == GGML_OP_VIEW;

    if (inplace && (tensor->src[0]->backend == GGML_BACKEND_GPU || tensor->src[0]->backend == GGML_BACKEND_GPU_SPLIT)) {
        ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu * ) tensor->src[0]->extra;
        char * src0_ddc = (char *) src0_extra->data_device[g_main_device];
        size_t view_offset = 0;
        if (tensor->op == GGML_OP_VIEW) {
            memcpy(&view_offset, tensor->op_params, sizeof(size_t));
        }
        extra->data_device[g_main_device] = src0_ddc + view_offset;
    } else {
        extra->data_device[g_main_device] = (char *) g_scratch_buffer + offset;
    }

    tensor->extra = extra;
}

void ggml_cann_copy_to_device(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor->backend == GGML_BACKEND_GPU);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) tensor->extra;
    ggml_cann_set_device(g_main_device);
    ACL_CHECK(aclrtMemcpy(extra->data_device[g_main_device], ggml_nbytes(tensor), tensor->data, ggml_nbytes(tensor), ACL_MEMCPY_HOST_TO_DEVICE));
}

void ggml_cann_copy_to_host(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor->backend != GGML_BACKEND_CPU && "cannot copy to host from CPU tensor");
    GGML_ASSERT(tensor->backend != GGML_BACKEND_GPU_SPLIT && "not implemented");

    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) tensor->extra;
    ggml_cann_set_device(g_main_device);

    // assumes GPU data is contiguous and CPU buffer is allocated
    ACL_CHECK(aclrtMemcpy(tensor->data, ggml_nbytes(tensor), extra->data_device[g_main_device], ggml_nbytes(tensor), ACL_MEMCPY_DEVICE_TO_HOST));
}

void ggml_cann_assign_buffers(struct ggml_tensor * tensor) {
    if (tensor == NULL)
        return;
    ggml_cann_assign_buffers_impl(tensor, true, false, false);
}

void ggml_cann_assign_buffers_no_alloc(struct ggml_tensor * tensor) {
    ggml_cann_assign_buffers_impl(tensor, true, false, true);
}

void ggml_cann_assign_buffers_no_scratch(struct ggml_tensor * tensor) {
    ggml_cann_assign_buffers_impl(tensor, false, false, false);
}

void ggml_cann_assign_buffers_force_inplace(struct ggml_tensor * tensor) {
    ggml_cann_assign_buffers_impl(tensor, false, true, false);
}

void ggml_cann_set_main_device(const int main_device) {
    if (main_device >= g_device_count) {
        fprintf(stderr, "warning: cannot set main_device=%d because there are only %d devices. Using device %d instead.\n",
                main_device, g_device_count, g_main_device);
        return;
    }
    printf("set main device to %d\n", main_device);
    g_main_device = main_device;
}

void ggml_cann_set_scratch_size(const size_t scratch_size) {
    // this is a hack to not completely break llama.cpp when using multiple models or contexts simultaneously
    // it still won't always work as expected, but it's better than nothing
    if (scratch_size > g_scratch_size) {
        ggml_cann_free_scratch();
    }
    g_scratch_size = std::max(g_scratch_size, scratch_size);
}

void ggml_cann_free_scratch() {
    if (g_scratch_buffer == nullptr) {
        return;
    }

    ACL_CHECK(aclrtFree(g_scratch_buffer));
    g_scratch_buffer = nullptr;
}


//#define DEBUG_CANN_MALLOC
/**
 * @brief A pool of CANN buffers(legacy).
 *
 * This class manages a pool of CANN buffers for a specific device.
 */
struct ggml_cann_pool_leg : public ggml_cann_pool {
    /**
     * @brief The maximum number of buffers in the pool.
     */
    static const int MAX_BUFFERS = 256;

    /**
     * @brief The device ID associated with this buffer pool.
     */
    int device;

    /**
     * @brief Structure representing a CANN buffer.
     */
    struct ggml_cann_buffer {
        void* ptr = nullptr;  ///< Pointer to the buffer memory.
        size_t size = 0;      ///< Size of the buffer.
    };

    /**
     * @brief Array of CANN buffers in the pool.
     */
    ggml_cann_buffer buffer_pool[MAX_BUFFERS] = {};

    /**
     * @brief Total size of all buffers in the pool.
     */
    size_t pool_size = 0;

    /**
     * @brief Constructor to initialize the buffer pool for a specific device.
     *
     * @param device The device ID to associate with this buffer pool.
     */
    explicit ggml_cann_pool_leg(int device) : device(device) {}

    /**
     * @brief Destructor to free all buffers in the pool.
     */
    ~ggml_cann_pool_leg() {
        ggml_cann_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cann_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
                ACL_CHECK(aclrtFree(b.ptr));
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
    }

    /**
     * @brief Allocate a buffer of the given size.
     *
     * @param size The size of the buffer to allocate.
     * @param actual_size A pointer to a variable to receive the actual size of
     * the allocated buffer.
     * @return A pointer to the allocated buffer.
     */
    void* alloc(size_t size, size_t* actual_size) override {
#ifdef DEBUG_CANN_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cann_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CANN_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void* ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cann_buffer& b = buffer_pool[ibest];
            void* ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void* ptr;
        size_t look_ahead_size = (size_t)(1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255) / 256);
        ggml_cann_set_device(device);
        ACL_CHECK(
            aclrtMalloc(&ptr, look_ahead_size, ACL_MEM_MALLOC_HUGE_FIRST));
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
#ifdef DEBUG_CANN_MALLOC
        GGML_LOG_INFO(
            "%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, "
            "requested %u MB\n",
            __func__, device, nnz, (uint32_t)(max_size / 1024 / 1024),
            (uint32_t)(pool_size / 1024 / 1024),
            (uint32_t)(size / 1024 / 1024));
#endif
        return ptr;
    }

    /**
     * @brief Free a buffer and return it to the pool.
     *
     * @param ptr Pointer to the buffer to free.
     * @param size Size of the buffer to free.
     */
    void free(void* ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cann_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        // memory should always buffered. these memory may still needed by
        // tasks in stream.
        // TODO, fix me.
        GGML_ASSERT(false && "Cann buffer pool full, increase MAX_CANN_BUFFERS\n");
    }
};

/**
 * @brief Create a new CANN pool for a specific device.
 *
 * Factory method to create a new CANN pool object based on the device type.
 *
 * @param device The device ID for which to create the pool.
 * @return A unique pointer to the created CANN pool.
 */
std::unique_ptr<ggml_cann_pool> ggml_backend_cann_context::new_pool_for_device(
    int device) {
    // return std::unique_ptr<ggml_cann_pool>(new ggml_cann_pool_leg(device));
    return std::unique_ptr<ggml_cann_pool>(new ggml_cann_pool_leg(device));
}


static void ggml_cann_nop(ggml_backend_cann_context& ctx, ggml_tensor * dst) {
    (void) ctx;
    (void) dst;
}

typedef void (*ggml_cann_func_t)(ggml_backend_cann_context& ctx, ggml_tensor * dst);
bool ggml_cann_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor){
    if (!g_cann_loaded) return false;

    ggml_cann_func_t func = nullptr;
    const bool src0_on_device = tensor->src[0] != nullptr && (tensor->src[0]->backend != GGML_BACKEND_CPU);
    const bool any_on_device = tensor->backend == GGML_BACKEND_GPU || src0_on_device
        || (tensor->src[1] != nullptr && tensor->src[1]->backend == GGML_BACKEND_GPU);

    // when src0 (weights) is not on device, we compute on CPU with sparsity
    if (!src0_on_device && (tensor->op == GGML_OP_MUL_MAT_SPARSE || tensor->op == GGML_OP_AXPY)
        || !any_on_device && tensor->op != GGML_OP_MUL_MAT) {
        return false;
    }
    if (!src0_on_device){
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT) {
        if (tensor->src[0]->ne[3] != tensor->src[1]->ne[3]) {
#ifndef NDEBUG
            fprintf(stderr, "%s: cannot compute %s: src0->ne[3] = %d, src1->ne[3] = %d - fallback to CPU\n", __func__, tensor->name, tensor->src[0]->ne[3], tensor->src[1]->ne[3]);
#endif
            return false;
        }
    }

    switch (tensor->op) {
        case GGML_OP_REPEAT:
            func = ggml_cann_repeat;
            break;
        case GGML_OP_GET_ROWS:
            func = ggml_cann_get_rows;
            break;
        case GGML_OP_DUP:
            func = ggml_cann_dup;
            break;
        case GGML_OP_ADD:
            func = ggml_cann_add;
            break;
        case GGML_OP_MUL:
            func = ggml_cann_mul_div<aclnnMulGetWorkspaceSize, aclnnMul>;
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(tensor)) {
                case GGML_UNARY_OP_GELU:
                    func = ggml_cann_activation<aclnnGeluGetWorkspaceSize, aclnnGelu>;
                    break;
                case GGML_UNARY_OP_SILU:
                    func = ggml_cann_activation<aclnnSiluGetWorkspaceSize, aclnnSilu>;
                    break;
                case GGML_UNARY_OP_RELU:
                    func = ggml_cann_activation<aclnnReluGetWorkspaceSize, aclnnRelu>;
                    break;
                default:
                    GGML_ASSERT(false && "not implemented op");
                    return false;
            } break;
        case GGML_OP_NORM:
            func = ggml_cann_norm;
            break;
        case GGML_OP_RMS_NORM:
            func = ggml_cann_rms_norm;
            break;
        case GGML_OP_MUL_MAT:
            if (!any_on_device && !ggml_cann_can_mul_mat(tensor->src[0], tensor->src[1], tensor)) {
                return false;
            }
            func = ggml_cann_mul_mat;
            break;
        case GGML_OP_MUL_MAT_SPARSE:
            if (!src0_on_device && !ggml_cann_can_mul_mat(tensor->src[0], tensor->src[1], tensor)) {
                return false;
            }
            func = ggml_cann_mul_mat_sparse;
            break;
        case GGML_OP_AXPY:
            func = ggml_cann_axpy;
            break;
        case GGML_OP_SCALE:
            func = ggml_cann_scale;
            break;
        case GGML_OP_SQR:
            func = ggml_cann_sqr;
            break;
        case GGML_OP_CLAMP:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cann_clamp;
            break;
        case GGML_OP_CPY:
            func = ggml_cann_cpy;
            break;
        case GGML_OP_CONT:
            func = ggml_cann_dup;
            break;
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
            func = ggml_cann_nop;
            break;
        case GGML_OP_DIAG_MASK_INF:
            break;
        case GGML_OP_SOFT_MAX:
            func = ggml_cann_softmax;
            break;
        case GGML_OP_ROPE:
            func = ggml_cann_rope;
            break;
        case GGML_OP_ALIBI:
            GGML_ASSERT(false && "not implemented this op, included in softmax op");
            break;
        case GGML_OP_IM2COL:
            func = ggml_cann_im2col;
            break;
        default:
            return false;
    }

    if (params->ith != 0) {
        return true;
    }
    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return true;
    }

    // printf("compute node %s, op: %d\n", tensor->name, tensor->op);
    if(func == nullptr){
        GGML_ASSERT(false && "not implemented this op");
    }

    ggml_tensor* src1 = tensor->src[1];
    ggml_tensor* idx = tensor->src[2];
    ggml_tensor* lst = tensor->src[3];
    ggml_cann_pool_alloc input_alloctor(g_ctx->pool());
    ggml_cann_pool_alloc output_alloctor(g_ctx->pool());
    ggml_cann_pool_alloc idx_alloctor(g_ctx->pool());
    ggml_cann_pool_alloc lst_alloctor(g_ctx->pool());
    if (src1 != nullptr && src1->backend == GGML_BACKEND_CPU) {
        // copy input to device
        GGML_ASSERT(src1->extra == nullptr);
        input_alloctor.alloc(ggml_nbytes(src1));
        void* input_buffer = input_alloctor.get();
        aclrtMemcpyAsync(input_buffer, ggml_nbytes(src1), src1->data, ggml_nbytes(src1), ACL_MEMCPY_HOST_TO_DEVICE, g_ctx->stream());

        ggml_tensor_extra_gpu* extra = (ggml_tensor_extra_gpu*)malloc(sizeof(ggml_tensor_extra_gpu));
        memset(extra, 0, sizeof(*extra));
        extra->data_device[g_ctx->device] = input_buffer;
        src1->extra = extra;
    }
    if (idx != nullptr && idx->backend == GGML_BACKEND_CPU) {
        // copy input to device
        GGML_ASSERT(idx->extra == nullptr);
        idx_alloctor.alloc(ggml_nbytes(idx));
        void* idx_buffer = idx_alloctor.get();
        aclrtMemcpyAsync(idx_buffer, ggml_nbytes(idx), idx->data, ggml_nbytes(idx), ACL_MEMCPY_HOST_TO_DEVICE, g_ctx->stream());

        ggml_tensor_extra_gpu* extra = (ggml_tensor_extra_gpu*)malloc(sizeof(ggml_tensor_extra_gpu));
        memset(extra, 0, sizeof(*extra));
        extra->data_device[g_ctx->device] = idx_buffer;
        idx->extra = extra;
    }
    if (lst != nullptr && lst->backend == GGML_BACKEND_CPU) {
        // copy input to device
        GGML_ASSERT(lst->extra == nullptr);
        lst_alloctor.alloc(ggml_nbytes(lst));
        void* lst_buffer = lst_alloctor.get();
        aclrtMemcpyAsync(lst_buffer, ggml_nbytes(lst), lst->data, ggml_nbytes(lst), ACL_MEMCPY_HOST_TO_DEVICE, g_ctx->stream());

        ggml_tensor_extra_gpu* extra = (ggml_tensor_extra_gpu*)malloc(sizeof(ggml_tensor_extra_gpu));
        memset(extra, 0, sizeof(*extra));
        extra->data_device[g_ctx->device] = lst_buffer;
        lst->extra = extra;
    }

    if(tensor->backend == GGML_BACKEND_CPU){
        // alloc output buffer
        GGML_ASSERT(tensor->extra == nullptr);
        output_alloctor.alloc(ggml_nbytes(tensor));
        void* output_buffer = output_alloctor.get();
        ggml_tensor_extra_gpu* extra = (ggml_tensor_extra_gpu*)malloc(sizeof(ggml_tensor_extra_gpu));
        memset(extra, 0, sizeof(*extra));
        extra->data_device[g_ctx->device] = output_buffer;
        tensor->extra = extra;
    }

    func(*g_ctx, tensor);

    if(src1 != nullptr && src1->backend == GGML_BACKEND_CPU){
        GGML_ASSERT(src1->extra != nullptr);
        free(src1->extra);
        src1->extra = nullptr;
    }
    if(idx != nullptr && idx->backend == GGML_BACKEND_CPU){
        GGML_ASSERT(idx->extra != nullptr);
        free(idx->extra);
        idx->extra = nullptr;
    }
    if(lst != nullptr && lst->backend == GGML_BACKEND_CPU){
        GGML_ASSERT(lst->extra != nullptr);
        free(lst->extra);
        lst->extra = nullptr;
    }
    if(tensor->backend == GGML_BACKEND_CPU){
        // copy output to host if needed
        void* output_buffer = ((ggml_tensor_extra_gpu*)tensor->extra)->data_device[g_ctx->device];
        aclrtMemcpyAsync(tensor->data, ggml_nbytes(tensor), output_buffer, ggml_nbytes(tensor), ACL_MEMCPY_DEVICE_TO_HOST, g_ctx->stream());
        aclrtSynchronizeStream(g_ctx->stream());

        GGML_ASSERT(tensor->extra != nullptr);
        free(tensor->extra);
        tensor->extra = nullptr;
    }

    return true;
}

size_t ggml_cann_get_free_memory(int device){
    ggml_cann_set_device(device);
    size_t free, total;
    ACL_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, &free, &total));
    return free;
}

void ggml_cann_set_device_constants(float sparse_pred_threshold){
    // fixed = 0
    return;
}

