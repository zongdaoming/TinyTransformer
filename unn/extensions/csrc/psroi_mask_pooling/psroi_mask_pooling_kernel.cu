#include "psroi_mask_pooling/psroi_mask_pooling.h"

using at::Tensor;
using at::Half;

#define DIVUP(m, n) ((m) / (m) + ((m) % (n) > 0))

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)


#ifndef CAFFE_COMMON_CUH_
#define CAFFE_COMMON_CUH_

#include <cuda.h>

  #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

  #else//performence loss
      static __inline__ __device__ double atomicAdd(double *address, double val) {
        unsigned long long int* address_as_ull = (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;
        if (val==0.0)
          return __longlong_as_double(old);
        do {
          assumed = old;
          old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
        } while (assumed != old);
        return __longlong_as_double(old);
      }
  #endif
#endif

static __inline__ __device__ at::Half atomicAdd(at::Half* address, at::Half val) {
  unsigned int *aligned = (unsigned int*)((size_t)address - ((size_t)address & 2));
  unsigned int old = *aligned;
  unsigned int assumed;
  unsigned short old_as_us;
  do {
    assumed = old;
    old_as_us = (unsigned short)((size_t)address & 2 ? old >> 16 : old & 0xffff);
#if __CUDACC_VER_MAJOR__ >= 9
    half sum = __float2half_rn(__half2float(__ushort_as_half(old_as_us)) + float(val));
    unsigned short sum_as_us = __half_as_ushort(sum);
#else
    unsigned short sum_as_us = __float2half_rn(__half2float(old_as_us) + float(val));
#endif
    unsigned int sum_as_ui = (size_t)address & 2 ? (sum_as_us << 16) | (old & 0xffff)
                                                 : (old & 0xffff0000) | sum_as_us;
    old = atomicCAS(aligned, assumed, sum_as_ui);
  } while(assumed != old);
  //__half_raw raw = {old_as_us};
  //return at::Half(raw);
  return at::Half({__ushort_as_half(old_as_us)});
};

template <typename scalar_t>
__global__ void PSROIMaskPoolingForward(
    const int nthreads,
    const scalar_t* bottom_data,
    const float spatial_scale,
    const float roi_scale,
    const float bin_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const scalar_t* bottom_rois,
    const int output_dim,
    const int group_size,
    scalar_t* top_data,
    int* mapping_channel,
    const int shape) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        // The output is in order (n, ctop, ph, pw)
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int ctop = (index / pooled_width / pooled_height) % output_dim;
        int n = index / pooled_width / pooled_height / output_dim;

        const scalar_t *rois = bottom_rois + n * shape;

        const int roi_batch_ind = static_cast<int>(rois[0]);

        const scalar_t x1 = rois[1];
        const scalar_t y1 = rois[2];
        const scalar_t x2 = rois[3];
        const scalar_t y2 = rois[4];

        scalar_t w = x2 - x1;
        scalar_t h = y2 - y1;

        scalar_t xc = (x1 + x2) * float(0.5);
        scalar_t yc = (y1 + y2) * float(0.5);

        // Rescale RoIs with regard to roi_scale
        scalar_t xx1 = xc - w * roi_scale * float(0.5);
        scalar_t xx2 = xc + w * roi_scale * float(0.5);
        scalar_t yy1 = yc - h * roi_scale * float(0.5);
        scalar_t yy2 = yc + h * roi_scale * float(0.5);

        scalar_t roi_start_w = round(xx1) * spatial_scale;
        scalar_t roi_start_h = round(yy1) * spatial_scale;
        scalar_t roi_end_w = (round(xx2) + float(1.)) * spatial_scale;
        scalar_t roi_end_h = (round(yy2) + float(1.)) * spatial_scale;

        // Force too small ROIs to be 1 x 1
        scalar_t roi_width = max(roi_end_w - roi_start_w, float(0.1));  // avoid 0
        scalar_t roi_height = max(roi_end_h - roi_start_h, float(0.1));

        // Compute w and h at bottom
        scalar_t bin_size_h = roi_height / static_cast<float>(pooled_height);
        scalar_t bin_size_w = roi_width / static_cast<float>(pooled_width);

        scalar_t delta_h = (bin_size_h * bin_scale - bin_size_h) * float(0.5);
        scalar_t delta_w = (bin_size_w * bin_scale - bin_size_w) * float(0.5);

        int hstart = static_cast<int>(
            floor((static_cast<float>(ph) * bin_size_h + roi_start_h) - delta_h));
        int wstart = static_cast<int>(
            floor((static_cast<float>(pw)* bin_size_w + roi_start_w) - delta_w));
        int hend = static_cast<int>(
            ceil((static_cast<float>(ph + 1) * bin_size_h + roi_start_h) + delta_h));
        int wend = static_cast<int>(
            ceil((static_cast<float>(pw + 1) * bin_size_w + roi_start_w) + delta_w));
        // Add roi offsets and clip to input boundaries
        hstart = min(max(hstart, 0), height);
        hend = min(max(hend, 0), height);
        wstart = min(max(wstart, 0), width);
        wend = min(max(wend, 0), width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        int gw = pw;
        int gh = ph;
        int c = (ctop * group_size + gh) * group_size + gw;

        const scalar_t *input = bottom_data + (roi_batch_ind * channels + c) * height * width;
        scalar_t out_sum = 0;
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            int bottom_index = h * width + w;
            out_sum += input[bottom_index];
          }
        }

        scalar_t bin_area = (hend - hstart) * (wend - wstart);
        top_data[index] = is_empty ? scalar_t(0.) : (out_sum / bin_area);
        mapping_channel[index] = c;
    }
  }


int PSROIMaskPoolForwardLaucher(
    at::Tensor bottom_data,
    const float spatial_scale, const float roi_scale, const float bin_scale,
    const int num_rois, const int output_dim, const int size_rois,
    const int height, const int width, const int channels,
    const int pooled_height, const int pooled_width,
    at::Tensor bottom_rois, at::Tensor top_data, at::Tensor mapping_channel) {
    const int kThreadsPerBlock = 1024;
    int output_size = num_rois * pooled_height * pooled_width * output_dim;
    cudaError_t err;

    err = cudaGetLastError();
    if(cudaSuccess != err) {
        fprintf( stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__,
                __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(bottom_data.type(),
                                        "psroi_mask_pooling_forward_cuda",
                                        ([&] {
        PSROIMaskPoolingForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
      output_size, bottom_data.data<scalar_t>(), spatial_scale, roi_scale, bin_scale, channels, height, width,
      pooled_height, pooled_width, bottom_rois.data<scalar_t>(), output_dim, pooled_height,
      top_data.data<scalar_t>(), mapping_channel.data<int>(), size_rois);
    }));
    // pooled_height == pooled_width == group_size
    err = cudaGetLastError();
    if(cudaSuccess != err) {
        fprintf( stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__,
                __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}

template <typename scalar_t>
__global__ void PSROIMaskPoolingBackward(
    const int nthreads,
    const scalar_t* top_diff,
    const int* mapping_channel,
    const int num_rois,
    const float spatial_scale,
    const float roi_scale,
    const float bin_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int output_dim,
    scalar_t* bottom_diff,
    const scalar_t* bottom_rois,
    const int shape) {
    CUDA_KERNEL_LOOP(index, nthreads) {
         // The output is in order (n, ctop, ph, pw)
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int n = index / pooled_width / pooled_height / output_dim;

        const scalar_t *rois = bottom_rois + n*shape;
        //bottom_rois += n * shape;

        const int roi_batch_ind = static_cast<int>(rois[0]);

        const scalar_t x1 = rois[1];
        const scalar_t y1 = rois[2];
        const scalar_t x2 = rois[3];
        const scalar_t y2 = rois[4];

        scalar_t w = x2 - x1;
        scalar_t h = y2 - y1;

        scalar_t xc = (x1 + x2) * float(0.5);
        scalar_t yc = (y1 + y2) * float(0.5);

        // Rescale RoIs with regard to roi_scale
        scalar_t xx1 = xc - w * roi_scale * float(0.5);
        scalar_t xx2 = xc + w * roi_scale * float(0.5);
        scalar_t yy1 = yc - h * roi_scale * float(0.5);
        scalar_t yy2 = yc + h * roi_scale * float(0.5);

        scalar_t roi_start_w = round(xx1) * spatial_scale;
        scalar_t roi_start_h = round(yy1) * spatial_scale;
        scalar_t roi_end_w = (round(xx2) + float(1.)) * spatial_scale;
        scalar_t roi_end_h = (round(yy2) + float(1.)) * spatial_scale;

        // Force too small ROIs to be 1 x 1
        scalar_t roi_width = max(roi_end_w - roi_start_w, float(0.1));  // avoid 0
        scalar_t roi_height = max(roi_end_h - roi_start_h, float(0.1));

        // Compute w and h at bottom
        scalar_t bin_size_h = roi_height / static_cast<float>(pooled_height);
        scalar_t bin_size_w = roi_width / static_cast<float>(pooled_width);

        scalar_t delta_h = (bin_size_h * bin_scale - bin_size_h) * float(0.5);
        scalar_t delta_w = (bin_size_w * bin_scale - bin_size_w) * float(0.5);

        int hstart = static_cast<int>(
            floor((static_cast<float>(ph) * bin_size_h + roi_start_h) - delta_h));
        int wstart = static_cast<int>(
            floor((static_cast<float>(pw)* bin_size_w + roi_start_w) - delta_w));
        int hend = static_cast<int>(
            ceil((static_cast<float>(ph + 1) * bin_size_h + roi_start_h) + delta_h));
        int wend = static_cast<int>(
            ceil((static_cast<float>(pw + 1) * bin_size_w + roi_start_w) + delta_w));
        // Add roi offsets and clip to input boundaries
        hstart = min(max(hstart, 0), height);
        hend = min(max(hend, 0), height);
        wstart = min(max(wstart, 0), width);
        wend = min(max(wend, 0), width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        // Compute c at bottom
        int c = mapping_channel[index];
        scalar_t* offset_bottom_diff = bottom_diff +
            (roi_batch_ind * channels + c) * height * width;
        scalar_t bin_area = (hend - hstart)*(wend - wstart);
        scalar_t diff_val = is_empty ? scalar_t(0.) : top_diff[index] / bin_area;
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            int bottom_index = h * width + w;
            // caffe_gpu_atomic_add(diff_val, offset_bottom_diff + bottom_index);
            atomicAdd(offset_bottom_diff + bottom_index, diff_val);
          }
        }
    }
  }

int PSROIMaskPoolBackwardLaucher(
    at::Tensor top_diff, const float spatial_scale,
    const float roi_scale, const float bin_scale, const int batch_size, const int num_rois,
    const int output_dim, const int size_rois, const int height, const int width, const int channels,
    const int pooled_height, const int pooled_width,
    at::Tensor bottom_rois, at::Tensor bottom_diff, at::Tensor mapping_channel) {
    const int kThreadsPerBlock = 1024;
    //int output_size = batch_size * height * width * output_dim;
    int output_size = output_dim * pooled_height * pooled_width * num_rois;
    cudaError_t err;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(top_diff.type(),
                                        "psroi_mask_pooling_backward_cuda",
                                        ([&] {
      PSROIMaskPoolingBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
      output_size, top_diff.data<scalar_t>(), mapping_channel.data<int>(), num_rois,
      spatial_scale, roi_scale, bin_scale, channels,
      height, width, pooled_height, pooled_width, output_dim, 
      bottom_diff.data<scalar_t>(), bottom_rois.data<scalar_t>(), size_rois);
    }));

    err = cudaGetLastError();
    if(cudaSuccess != err) {
        fprintf( stderr, "%s#%d: cudaCheckError() failed : %s\n", __FILE__,
                __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}

