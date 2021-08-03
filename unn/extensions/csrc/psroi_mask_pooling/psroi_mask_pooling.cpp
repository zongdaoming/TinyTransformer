#include "psroi_mask_pooling/psroi_mask_pooling.h"

using at::Tensor;

int psroi_mask_pooling_forward(int pooled_height,
                               int pooled_width,
                               int output_dim,
                               float spatial_scale,
                               float roi_scale,
                               float bin_scale,
                               at::Tensor features,
                               at::Tensor rois,
                               at::Tensor output,
                               at::Tensor mapping_channel) {
  // ONNX requires cpu forward
  return 0;
}

int psroi_mask_pooling_forward_cuda(int pooled_height,
                                    int pooled_width,
                                    int output_dim,
                                    float spatial_scale,
                                    float roi_scale,
                                    float bin_scale,
                                    at::Tensor features,
                                    at::Tensor rois,
                                    at::Tensor output,
                                    at::Tensor mapping_channel) {
    CHECK_INPUT(features);
    CHECK_INPUT(rois);
    CHECK_INPUT(output);
    CHECK_INPUT(mapping_channel);
    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);

    AT_ASSERTM(size_rois == 5, "rois shape is expected to be N * 5");
    int feat_height = features.size(2);
    int feat_width = features.size(3);
    int num_channels = features.size(1);

    PSROIMaskPoolForwardLaucher(
        features,
        spatial_scale, roi_scale, bin_scale,
        num_rois, output_dim, size_rois,
        feat_height, feat_width, num_channels,
        pooled_height, pooled_width,
        rois, output, mapping_channel);

    return 0;
}

int psroi_mask_pooling_backward_cuda(int pooled_height,
                                     int pooled_width,
                                     int output_dim,
                                     float spatial_scale,
                                     float roi_scale,
                                     float bin_scale,
                                     at::Tensor top_grad,
                                     at::Tensor rois,
                                     at::Tensor bottom_grad,
                                     at::Tensor mapping_channel) {
    CHECK_INPUT(top_grad);
    CHECK_INPUT(rois);
    CHECK_INPUT(bottom_grad);
    CHECK_INPUT(mapping_channel);

    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);

    AT_ASSERTM(size_rois == 5, "rois shape is expected to be N * 5");
    int batch_size = bottom_grad.size(0);
    int feat_height = bottom_grad.size(2);
    int feat_width = bottom_grad.size(3);
    int num_channels = bottom_grad.size(1);

    PSROIMaskPoolBackwardLaucher(
        top_grad,
        spatial_scale, roi_scale, bin_scale,
        batch_size, num_rois, output_dim, size_rois,
        feat_height, feat_width, num_channels,
        pooled_height, pooled_width,
        rois, bottom_grad, mapping_channel);

    return 0;
}
