#include "roi_pooling/roi_pooling.h"

using at::Tensor;

int roi_pooling_forward_cuda(
    int pooled_height,
    int pooled_width,
    float spatial_scale,
    Tensor features,
    Tensor rois,
    Tensor output,
    Tensor argmax)
{
    // Grab the input tensor
    CHECK_INPUT(features);
    CHECK_INPUT(rois);
    CHECK_INPUT(argmax);
    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);
    if (size_rois != 5)
    {
        exit(1);
        return 1;
    }
    // data height
    int data_height = features.size(2);
    // data width
    int data_width = features.size(3);
    // Number of channels
    int num_channels = features.size(1);

    ROIPoolForwardLaucher(
        features, spatial_scale, num_rois, data_height,
        data_width, num_channels, pooled_height,
        pooled_width, rois, output, argmax);
    return 0;
}

int roi_pooling_backward_cuda(
    int pooled_height, 
    int pooled_width, 
    float spatial_scale,
    Tensor  top_grad,
    Tensor  rois,
    Tensor bottom_grad,
    Tensor  argmax)
{
    // Grab the input tensor
    CHECK_INPUT(top_grad);
    CHECK_INPUT(rois);
    CHECK_INPUT(argmax);
    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);
    if (size_rois != 5)
    {
        exit(1);
        return 1;
    }

    int batch_size = bottom_grad.size(0);
    int num_channels = bottom_grad.size(1);
    int data_height = bottom_grad.size(2); 
    int data_width = bottom_grad.size(3);
    ROIPoolBackwardLaucher(
        top_grad, spatial_scale, 
        batch_size, num_rois, data_height, data_width, 
        num_channels, pooled_height, pooled_width, 
        rois,
        bottom_grad,
        argmax);
    return 0;
}
