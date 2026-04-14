#include "pool1d.cuh"

static __global__ void pool1d_kernel_f32(
        const float * src,
        float * dst,
        const int64_t src_nb1,
        const int64_t src_nb2,
        const int64_t src_nb3,
        const int64_t src_ne1,
        const int64_t src_ne2,
        const int64_t dst_stride,
        const int64_t iw,
        const int64_t ow,
        const int64_t np,
        const int k0,
        const int s0,
        const int p0,
        const enum ggml_op_pool op) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= np) {
        return;
    }

    const int64_t out_col = idx % ow;
    const int64_t row = idx / ow;
    const int base = (int) out_col * s0 - p0;

    float acc = op == GGML_OP_POOL_MAX ? -FLT_MAX : 0.0f;
    int count = 0;

    const int64_t rows_per_plane = src_ne1 * src_ne2;
    const int64_t i3 = rows_per_plane > 0 ? row / rows_per_plane : 0;
    const int64_t row_in_plane = rows_per_plane > 0 ? row % rows_per_plane : 0;
    const int64_t i2 = src_ne1 > 0 ? row_in_plane / src_ne1 : 0;
    const int64_t i1 = src_ne1 > 0 ? row_in_plane % src_ne1 : 0;

    const float * src_row = src + i1 * src_nb1 + i2 * src_nb2 + i3 * src_nb3;
    for (int ki = 0; ki < k0; ++ki) {
        const int j = base + ki;
        if (j < 0 || j >= iw) {
            continue;
        }

        const float value = src_row[j];
        if (op == GGML_OP_POOL_MAX) {
            acc = fmaxf(acc, value);
        } else {
            acc += value;
            ++count;
        }
    }

    if (op == GGML_OP_POOL_AVG) {
        acc = count > 0 ? acc / (float) count : 0.0f;
    }

    dst[row * dst_stride + out_col] = acc;
}

static __global__ void pool1d_kernel_f16(
        const half * src,
        float * dst,
    const int64_t src_nb1,
    const int64_t src_nb2,
    const int64_t src_nb3,
    const int64_t src_ne1,
    const int64_t src_ne2,
        const int64_t dst_stride,
        const int64_t iw,
        const int64_t ow,
        const int64_t np,
        const int k0,
        const int s0,
        const int p0,
        const enum ggml_op_pool op) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= np) {
        return;
    }

    const int64_t out_col = idx % ow;
    const int64_t row = idx / ow;
    const int base = (int) out_col * s0 - p0;

    float acc = op == GGML_OP_POOL_MAX ? -FLT_MAX : 0.0f;
    int count = 0;

    const int64_t rows_per_plane = src_ne1 * src_ne2;
    const int64_t i3 = rows_per_plane > 0 ? row / rows_per_plane : 0;
    const int64_t row_in_plane = rows_per_plane > 0 ? row % rows_per_plane : 0;
    const int64_t i2 = src_ne1 > 0 ? row_in_plane / src_ne1 : 0;
    const int64_t i1 = src_ne1 > 0 ? row_in_plane % src_ne1 : 0;

    const half * src_row = src + i1 * src_nb1 + i2 * src_nb2 + i3 * src_nb3;
    for (int ki = 0; ki < k0; ++ki) {
        const int j = base + ki;
        if (j < 0 || j >= iw) {
            continue;
        }

        const float value = __half2float(src_row[j]);
        if (op == GGML_OP_POOL_MAX) {
            acc = fmaxf(acc, value);
        } else {
            acc += value;
            ++count;
        }
    }

    if (op == GGML_OP_POOL_AVG) {
        acc = count > 0 ? acc / (float) count : 0.0f;
    }

    dst[row * dst_stride + out_col] = acc;
}

void ggml_cuda_op_pool1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(dst->nb[0] == sizeof(float));

    const int32_t * opts = (const int32_t *) dst->op_params;
    const enum ggml_op_pool op = (enum ggml_op_pool) opts[0];
    const int k0 = opts[1];
    const int s0 = opts[2];
    const int p0 = opts[3];

    GGML_ASSERT(op == GGML_OP_POOL_AVG || op == GGML_OP_POOL_MAX);

    const int64_t iw = src0->ne[0];
    const int64_t ow = dst->ne[0];
    const int64_t np = ggml_nelements(dst);
    const int64_t src_nb1 = src0->nb[1] / ggml_type_size(src0->type);
    const int64_t src_nb2 = src0->nb[2] / ggml_type_size(src0->type);
    const int64_t src_nb3 = src0->nb[3] / ggml_type_size(src0->type);
    const int64_t src_ne1 = src0->ne[1];
    const int64_t src_ne2 = src0->ne[2];
    const int64_t dst_stride = dst->nb[1] / sizeof(float);

    const dim3 block_dims(CUDA_POOL1D_BLOCK_SIZE, 1, 1);
    const dim3 block_nums((np + CUDA_POOL1D_BLOCK_SIZE - 1) / CUDA_POOL1D_BLOCK_SIZE, 1, 1);

    if (src0->type == GGML_TYPE_F16) {
        pool1d_kernel_f16<<<block_nums, block_dims, 0, stream>>>(
                (const half *) src0->data,
                (float *) dst->data,
                src_nb1,
                src_nb2,
                src_nb3,
                src_ne1,
                src_ne2,
                dst_stride,
                iw,
                ow,
                np,
                k0,
                s0,
                p0,
                op);
    } else {
        pool1d_kernel_f32<<<block_nums, block_dims, 0, stream>>>(
                (const float *) src0->data,
                (float *) dst->data,
                src_nb1,
                src_nb2,
                src_nb3,
                src_ne1,
                src_ne2,
                dst_stride,
                iw,
                ow,
                np,
                k0,
                s0,
                p0,
                op);
    }
}