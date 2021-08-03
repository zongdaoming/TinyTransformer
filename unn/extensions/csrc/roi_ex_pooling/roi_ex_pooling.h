#ifndef ROIEXPOOLING_H_
#define ROIEXPOOLING_H_

#include <ATen/ATen.h>
#include <vector>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// for tocaffe support
int roi_ex_pooling_forward(int pooled_height,
                        int pooled_width,
                        float spatial_scale,
                        at::Tensor features,
                        at::Tensor rois,
                        at::Tensor output,
                        at::Tensor argmax);

int roi_ex_pooling_forward_cuda(
    int pooled_height,
    int pooled_width,
    float spatial_scale,
    at::Tensor features,
    at::Tensor rois,
    at::Tensor output,
    at::Tensor argmax);

int roi_ex_pooling_backward_cuda(
    int pooled_height, 
    int pooled_width, 
    float spatial_scale,
    at::Tensor  top_grad,
    at::Tensor  rois,
    at::Tensor bottom_grad,
    at::Tensor  argmax);

int ROIExPoolForwardLaucher(
    at::Tensor bottom_data, const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, at::Tensor bottom_rois,
    at::Tensor top_data, at::Tensor argmax_data);

int ROIExPoolBackwardLaucher(at::Tensor top_diff, const float spatial_scale, const int batch_size, const int num_rois,
    const int height, const int width, const int channels, const int pooled_height,
    const int pooled_width, at::Tensor bottom_rois,
    at::Tensor bottom_diff, at::Tensor argmax_data);

#endif
