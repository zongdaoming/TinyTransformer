#include "roi_ex_pooling/roi_ex_pooling.h"

using at::Tensor;

int roi_ex_pooling_forward(int pooled_height,
                        int pooled_width,
                        float spatial_scale,
                        Tensor features,
                        Tensor rois,
                        Tensor output,
                        Tensor argmax) {
    // ONNX requires cpu forward support
    return 0;
}
