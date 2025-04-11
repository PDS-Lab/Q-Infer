#include "kernel_operator.h"

// optimize me. Use template to avoid copy code.
using namespace AscendC;

#define BUFFER_NUM 2

// input float32(n_ff, n_batch)
// weight_gpu float16(n_embd, n_ff_gpu)
// output float32(n_embd, n_batch)
// gpu_bucket int32(n_ff_gpu)
// sparse_idx float32(n_ff, n_batch)


// axpy output[j][0:n_embd] += weight[i][0:n_embd] * input[j][i] 
class SPARSE_AXPY_f16 {
   public:
    __aicore__ inline SPARSE_AXPY_f16() {}
    __aicore__ inline void init(GM_ADDR input, GM_ADDR weight, GM_ADDR output, GM_ADDR lst, GM_ADDR idx,
                                int64_t *input_ne_ub, size_t *input_nb_ub,
                                int64_t *weight_ne_ub, size_t *weight_nb_ub,
                                int64_t *output_ne_ub, size_t *output_nb_ub) {
        // printf("SPARSE_AXPY_f16 init\n");
        // PRINTF("SPARSE_AXPY_f16 init\n");
        // TODO, use template for F16/f32
        int64_t op_block_num = GetBlockNum();
        int64_t op_block_idx = GetBlockIdx();

        for (int i = 0; i < 4; i++) {
            input_ne[i] = input_ne_ub[i];
            input_stride[i] = input_nb_ub[i] / input_nb_ub[0];

            weight_ne[i] = weight_ne_ub[i];
            weight_stride[i] = weight_nb_ub[i] / weight_nb_ub[0];

            output_ne[i] = output_ne_ub[i];
            output_stride[i] = output_nb_ub[i] / output_nb_ub[0];
        }

        // seperate neurons to different ai core.
        // n_rows = all rows should get.
        // dr, all rows should this thread get.
        // n_ff_gpu
        uint64_t n_rows = weight_ne[1];
        dr = n_rows / op_block_num;
        ir = dr * op_block_idx;

        batch_size = input_ne[1];

        input_gm.SetGlobalBuffer((__gm__ float *)input);
        weight_gm.SetGlobalBuffer((__gm__ half *)weight);
        output_gm.SetGlobalBuffer((__gm__ float *)output);
        lst_gm.SetGlobalBuffer((__gm__ int32_t *)lst);
        idx_gm.SetGlobalBuffer((__gm__ float *)idx);

        uint64_t weight_local_buffer_size = weight_ne[0] * sizeof(half);
        uint64_t output_local_buffer_size = weight_ne[0] * sizeof(float);

        pipe.InitBuffer(weight_queue, BUFFER_NUM, weight_local_buffer_size);
        pipe.InitBuffer(output_queue, BUFFER_NUM, output_local_buffer_size);
    }

    __aicore__ inline void copy_in(uint32_t input_offset, size_t input_length, uint32_t weight_offset, size_t weight_length) {
        LocalTensor<half> weight_local = weight_queue.AllocTensor<half>();
        DataCopy(weight_local, weight_gm[weight_offset], weight_length);
        weight_queue.EnQue(weight_local);
    }

    __aicore__ inline void copy_out(uint32_t offset, size_t length) {
        LocalTensor<float> output_local = output_queue.DeQue<float>();
        SetAtomicAdd<float>();
        DataCopy(output_gm[offset], output_local, length);
        SetAtomicNone();
        output_queue.FreeTensor(output_local);
    }

    __aicore__ inline void calculate_row(int64_t row_idx, int64_t batch_idx) {
        // optimize caclulate_row, by calculate more rows in one time.

        // 原权重矩阵的行索引
        const int32_t origin_row_idx = lst_gm.GetValue(row_idx);
        const int64_t input_offset = origin_row_idx * input_stride[0]
                                    + batch_idx * input_stride[1];
        const int64_t weight_offset = row_idx * weight_stride[1];
        const int64_t output_offset = batch_idx * output_stride[1];

        copy_in(input_offset, input_ne[0], weight_offset, weight_ne[0]);

        LocalTensor<half> weight_local = weight_queue.DeQue<half>();
        LocalTensor<float> output_local = output_queue.AllocTensor<float>();

        Cast(output_local, weight_local, RoundMode::CAST_NONE, weight_ne[0]);
        float scale = input_gm.GetValue(input_offset);
        Muls(output_local, output_local, scale, weight_ne[0]);

        output_queue.EnQue(output_local);
        copy_out(output_offset, weight_ne[0]);

        weight_queue.FreeTensor(weight_local);
    }

    __aicore__ inline void calculate() {
        for (int64_t i = ir; i < ir + dr; i++) {
            for (int64_t j = 0; j < batch_size; j++) {
                const int32_t origin_row_idx = lst_gm.GetValue(i);
                const int64_t sparse_idx_offset = origin_row_idx * input_stride[0] + j * input_stride[1];
                const float row_pred_active = idx_gm.GetValue(sparse_idx_offset);
                if (row_pred_active < 0) {
                    continue;
                }
                calculate_row(i, j);
            }
        }
    }

   private:
    int64_t input_ne[4];
    size_t input_stride[4];

    int64_t weight_ne[4];
    size_t weight_stride[4];

    int64_t output_ne[4];
    size_t output_stride[4];

    int64_t ir;
    int64_t dr;

    int64_t batch_size;

    TPipe pipe;
    GlobalTensor<float> input_gm;
    GlobalTensor<half> weight_gm;
    GlobalTensor<float> output_gm;
    GlobalTensor<int32_t> lst_gm;
    GlobalTensor<float> idx_gm;

    TQue<QuePosition::VECIN, BUFFER_NUM> weight_queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> output_queue;
};

template <typename T>
__aicore__ inline void copy_to_ub(GM_ADDR gm, T *ub, size_t size) {
    auto gm_ptr = (__gm__ uint8_t *)gm;
    auto ub_ptr = (uint8_t *)(ub);
    for (int32_t i = 0; i < size; ++i, ++ub_ptr, ++gm_ptr) {
        *ub_ptr = *gm_ptr;
    }
}

extern "C" __global__ __aicore__ void ascendc_sparse_axpy_f16(
    GM_ADDR input_gm, GM_ADDR weight_gm, GM_ADDR output_gm, GM_ADDR lst_gm, GM_ADDR idx_gm,
    GM_ADDR input_ne_gm, GM_ADDR input_nb_gm,
    GM_ADDR weight_ne_gm, GM_ADDR weight_nb_gm, 
    GM_ADDR output_ne_gm, GM_ADDR output_nb_gm) {
    // printf("SPARSE_AXPY_f16 func\n");
    // PRINTF("SPARSE_AXPY_f16 func\n");
    int64_t input_ne_ub[4];
    size_t input_nb_ub[4];
    int64_t weight_ne_ub[4];
    size_t weight_nb_ub[4];
    int64_t output_ne_ub[4];
    size_t output_nb_ub[4];

    copy_to_ub(input_ne_gm, input_ne_ub, 32);
    copy_to_ub(input_nb_gm, input_nb_ub, 32);
    copy_to_ub(weight_ne_gm, weight_ne_ub, 32);
    copy_to_ub(weight_nb_gm, weight_nb_ub, 32);
    copy_to_ub(output_ne_gm, output_ne_ub, 32);
    copy_to_ub(output_nb_gm, output_nb_ub, 32);

    SPARSE_AXPY_f16 op;
    op.init(input_gm, weight_gm, output_gm, lst_gm, idx_gm,
            input_ne_ub, input_nb_ub,
            weight_ne_ub, weight_nb_ub, 
            output_ne_ub, output_nb_ub);
    op.calculate();
}