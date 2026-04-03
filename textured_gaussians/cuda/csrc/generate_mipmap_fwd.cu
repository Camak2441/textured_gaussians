#include "bindings.h"
#include "helpers.cuh"
#include "types.cuh"
#include "filters/mip2.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/TensorAccessor.h>

namespace gsplat
{
    namespace cg = cooperative_groups;

    /**
     *
     */
    template <uint32_t COLOR_DIM, typename S>
    __global__ void generate_mipmap_fwd_kernel(
        const uint32_t tile_size,
        const uint32_t log_reduce,
        const uint32_t log_start_res,

        // outputs
        at::PackedTensorAccessor32<S, 3, at::RestrictPtrTraits> mip_textures // [N, Texels, 4]
    )
    {
        // Each thread draws one pixel of the image

        auto block = cg::this_thread_block();
        uint32_t g = block.group_index().x;
        uint32_t pixel_y = block.group_index().y * tile_size + block.thread_index().y;
        uint32_t pixel_x = block.group_index().z * tile_size + block.thread_index().z;

        uint32_t level_scale = 1 << log_start_res;
        uint32_t reduce_scale = 1 << log_reduce;

        for (int log_scale = log_start_res; log_scale > 0 && log_scale > log_start_res - log_reduce; --log_scale)
        {
            level_scale >>= 1;
            reduce_scale >>= 1;
            uint32_t t_low_base = t_base_index(log_scale, 0);
            uint32_t t_high_base = t_base_index(log_scale - 1, 0);
            for (int v = pixel_y * reduce_scale; v < (pixel_y + 1) * reduce_scale; ++v)
            {
                for (int u = pixel_x * reduce_scale; u < (pixel_x + 1) * reduce_scale; ++u)
                {
                    GSPLAT_PRAGMA_UNROLL
                    for (int k = 0; k < COLOR_DIM; ++k)
                    {
                        mip_textures[g][t_high_base + v * level_scale + u][k] =
                            (mip_textures[g][t_low_base + v * 2 * level_scale * 2 + u * 2][k] + mip_textures[g][t_low_base + v * 2 * level_scale * 2 + u * 2 + 1][k] + mip_textures[g][t_low_base + (v * 2 + 1) * level_scale * 2 + u * 2][k] + mip_textures[g][t_low_base + (v * 2 + 1) * level_scale * 2 + u * 2 + 1][k]) / 4.0f;
                    }
                }
            }
        }
    }

    template <uint32_t CDIM>
    torch::Tensor
    call_fwd_genmip_kernel_with_dim(
        // Gaussian parameters
        const torch::Tensor &textures,  //
        const uint32_t log_texture_res, //
        const uint32_t log_reduce,
        const uint32_t tile_size)
    {
        GSPLAT_CHECK_INPUT(textures);

        uint32_t N = textures.size(0); // number of gaussians
        uint32_t texture_res = 1 << log_texture_res;
        uint32_t channels = textures.size(3);

        uint32_t added_texels = t_base_index_host(log_texture_res, 0);

        torch::Tensor mip_textures = torch::cat({torch::empty(
                                                     {N, added_texels, channels},
                                                     textures.options()),
                                                 textures.reshape({N, texture_res * texture_res, channels})},
                                                1);

        // Each block covers a tile on the image. In total there are
        // C * tile_height * tile_width blocks.
        // we assign one pixel to one thread.
        uint32_t reduced_texture_res = 1 << log_texture_res;

        dim3 threads;
        dim3 blocks;
        for (int log_res = log_texture_res; log_res > 0; log_res -= log_reduce)
        {
            reduced_texture_res >>= log_reduce;
            if (log_res > log_reduce)
            {
                if (reduced_texture_res < tile_size)
                {
                    threads = {1, reduced_texture_res, reduced_texture_res};
                    blocks = {N, 1, 1};
                }
                else
                {
                    threads = {1, tile_size, tile_size};
                    blocks = {N, (reduced_texture_res + tile_size - 1) / tile_size, (reduced_texture_res + tile_size - 1) / tile_size};
                }

                at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
                generate_mipmap_fwd_kernel<CDIM, float>
                    <<<blocks, threads, 0, stream>>>(
                        tile_size,
                        log_reduce,
                        log_res,
                        mip_textures.packed_accessor32<float, 3, at::RestrictPtrTraits>());
            }
            else
            {
                threads = {1, 1, 1};
                blocks = {N, 1, 1};

                at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
                generate_mipmap_fwd_kernel<CDIM, float>
                    <<<blocks, threads, 0, stream>>>(
                        tile_size,
                        log_res,
                        log_res,
                        mip_textures.packed_accessor32<float, 3, at::RestrictPtrTraits>());
            }
        }

        return mip_textures;
    }

    torch::Tensor
    generate_mipmap_fwd_tensor(
        // Gaussian parameters
        const torch::Tensor &textures, //
        // tile size
        const uint32_t log_texture_res, //
        const uint32_t log_reduce,
        const uint32_t tile_size)
    {
        uint32_t channels = textures.size(-1);

#define __GS__CALL_(N)                             \
    case N:                                        \
        return call_fwd_genmip_kernel_with_dim<N>( \
            textures,                              \
            log_texture_res,                       \
            log_reduce,                            \
            tile_size);
        // TODO: an optimization can be done by passing the actual number of
        // channels into the kernel functions and avoid necessary global memory
        // writes. This requires moving the channel padding from python to C side.
        switch (channels)
        {
            __GS__CALL_(1)
            __GS__CALL_(2)
            __GS__CALL_(3)
            __GS__CALL_(4)
            __GS__CALL_(5)
            // __GS__CALL_(8)
            // __GS__CALL_(9)
            // __GS__CALL_(16)
            // __GS__CALL_(17)
            // __GS__CALL_(32)
            // __GS__CALL_(33)
            // __GS__CALL_(64)
            // __GS__CALL_(65)
            // __GS__CALL_(128)
            // __GS__CALL_(129)
            // __GS__CALL_(256)
            // __GS__CALL_(257)
            // __GS__CALL_(512)
            // __GS__CALL_(513)
        default:
            AT_ERROR("Unsupported number of channels: ", channels);
        }
    }
} // namespace gsplat
