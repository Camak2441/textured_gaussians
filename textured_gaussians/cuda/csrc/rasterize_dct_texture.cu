#include "bindings.h"
#include "helpers.cuh"
#include "types.cuh"
#include "filters/dct.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/TensorAccessor.h>

namespace gsplat
{

    namespace cg = cooperative_groups;

    /****************************************************************************
     * Rasterization of DCT textures
     ****************************************************************************/

    /**
     *
     */
    template <uint32_t COLOR_DIM, typename S>
    __global__ void rasterize_dct_textures_kernel(
        at::PackedTensorAccessor32<const S, 4, at::RestrictPtrTraits> textures, // [N, Texture_Resolution, Texture_Resolution, 4]
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,

        // outputs
        S *__restrict__ render_colors // [C, image_height, image_width, COLOR_DIM]
    )
    {
        // Each thread draws one pixel of the image

        auto block = cg::this_thread_block();
        uint32_t g = block.group_index().x;
        uint32_t pixel_y = block.group_index().y * tile_size + block.thread_index().y;
        uint32_t pixel_x = block.group_index().z * tile_size + block.thread_index().z;
        uint32_t pix_id = pixel_y * image_width + pixel_x;

        int texture_res_y = textures.size(1);
        int texture_res_x = textures.size(2);

        render_colors += g * image_height * image_width * COLOR_DIM + pix_id * COLOR_DIM;

        if (pixel_x >= image_width || pixel_y >= image_height)
        {
            return;
        }

        S u = ((S)pixel_x + 0.5f) / (image_width - 1);
        S v = ((S)pixel_y + 0.5f) / (image_height - 1);

        GSPLAT_PRAGMA_UNROLL
        for (uint32_t k = 0; k < COLOR_DIM; ++k)
        {
            S tex_color = dct_sample(textures, texture_res_x, texture_res_y, g, u, v, k);
            render_colors[k] = tex_color;
        }
    }

    template <uint32_t CDIM>
    torch::Tensor
    call_dct_kernel_with_dim(
        // Gaussian parameters
        const torch::Tensor &textures, //
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size)
    {
        GSPLAT_CHECK_INPUT(textures);

        uint32_t N = textures.size(0); // number of gaussians
        uint32_t channels = textures.size(-1);

        // Each block covers a tile on the image. In total there are
        // C * tile_height * tile_width blocks.
        // we assign one pixel to one thread.
        dim3 threads = {1, tile_size, tile_size};
        dim3 blocks = {N, (image_height + tile_size - 1) / tile_size, (image_width + tile_size - 1) / tile_size};

        torch::Tensor renders = torch::empty(
            {N, image_height, image_width, channels},
            textures.options().dtype(torch::kFloat32));

        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        rasterize_dct_textures_kernel<CDIM, float>
            <<<blocks, threads, 0, stream>>>(
                textures.packed_accessor32<const float, 4, at::RestrictPtrTraits>(),
                image_width,
                image_height,
                tile_size,
                renders.data_ptr<float>());

        return renders;
    }

    torch::Tensor
    rasterize_dct_textures_tensor(
        // Gaussian parameters
        const torch::Tensor &textures, //
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size)
    {
        uint32_t channels = textures.size(-1);

#define __GS__CALL_(N)                      \
    case N:                                 \
        return call_dct_kernel_with_dim<N>( \
            textures,                       \
            image_width,                    \
            image_height,                   \
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