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

#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif


#define GGML_CANN_NAME "CANN"
#define GGML_CANN_MAX_DEVICES 16


// Always success. To check if cann is actually loaded, use `ggml_cann_loaded`.
GGML_API void   ggml_init_cann(void);
// Returns `true` if there are available CANN devices and cann loads successfully; otherwise, it returns `false`.
GGML_API bool   ggml_cann_loaded(void);

GGML_API void * ggml_cann_host_malloc(size_t size);
GGML_API void   ggml_cann_host_free(void * ptr);

GGML_API bool   ggml_cann_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
GGML_API void   ggml_cann_set_tensor_split(const float * tensor_split);

GGML_API void   ggml_cann_transform_tensor(void * data, struct ggml_tensor * tensor);
GGML_API void   ggml_cann_alloc_tensor(struct ggml_tensor * tensor);
GGML_API void   ggml_cann_free_data(struct ggml_tensor * tensor);

GGML_API void   ggml_cann_stream_synchronize(const int stream);
GGML_API void   ggml_cann_cpy_ptr_1d(struct ggml_tensor *dst, const struct ggml_tensor *src, int host_i, int device_i);
GGML_API void   ggml_cann_cpy_1d(struct ggml_tensor *dst, const struct ggml_tensor *src);
GGML_API void **ggml_cann_get_data_pp(struct ggml_tensor * tensor);

GGML_API void   ggml_cann_assign_scratch_offset(struct ggml_tensor * tensor, size_t offset);
GGML_API void   ggml_cann_copy_to_device(struct ggml_tensor * tensor);
GGML_API void   ggml_cann_copy_to_host(struct ggml_tensor * tensor);

GGML_API void   ggml_cann_assign_buffers(struct ggml_tensor * tensor);
GGML_API void   ggml_cann_assign_buffers_no_alloc(struct ggml_tensor * tensor);
GGML_API void   ggml_cann_assign_buffers_no_scratch(struct ggml_tensor * tensor);
GGML_API void   ggml_cann_assign_buffers_force_inplace(struct ggml_tensor * tensor);

GGML_API void   ggml_cann_set_main_device(int main_device);
GGML_API void   ggml_cann_set_scratch_size(size_t scratch_size);
GGML_API void   ggml_cann_free_scratch(void);
GGML_API bool   ggml_cann_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor);

GGML_API size_t ggml_cann_get_free_memory(int device);

GGML_API void   ggml_cann_set_device_constants(float sparse_pred_threshold);


#ifdef __cplusplus
}
#endif
