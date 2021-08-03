#include "deform_conv/deformable_conv.h"
#include "focal_loss/focal_loss.h"
#include "psroi_pooling/psroi_pooling.h"
#include "psroi_mask_pooling/psroi_mask_pooling.h"
#include "roi_pooling/roi_pooling.h"
#include "roi_align/roi_align.h"
#include "nms/nms.h"
#include "iou_overlap/iou_overlap.h"

#include <torch/torch.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  // pybind deformable conv v1
  pybind11::module dc_v1 = m.def_submodule("deform_conv_v1",
                                           "deformable convolution v1");
  dc_v1.def("backward_parameters_cuda",
            &deform_conv_backward_parameters_cuda,
            "deform_conv_backward_parameters_cuda (CUDA)");
  dc_v1.def("backward_input_cuda",
            &deform_conv_backward_input_cuda,
            "deform_conv_backward_input_cuda (CUDA)");
  dc_v1.def("forward_cuda",
            &deform_conv_forward_cuda,
            "deform_conv_forward_cuda (CUDA)");
  dc_v1.def("forward_cpu",
            &deform_conv_forward,
            "deform_conv_forward_cpu (CPU)");

  // pybind focal loss
  pybind11::module fl = m.def_submodule("focal_loss",
                                        "focal loss for RetinaNet");
  fl.def("sigmoid_forward_cuda",
         &focal_loss_sigmoid_forward_cuda,
         "sigmoid_forward_cuda forward (CUDA)");
  fl.def("sigmoid_backward_cuda",
         &focal_loss_sigmoid_backward_cuda,
         "sigmoid_backward_cuda backward (CUDA)");
  fl.def("softmax_forward_cuda",
         &focal_loss_softmax_forward_cuda,
         "softmax_forward_cuda forward (CUDA)");
  fl.def("softmax_backward_cuda",
         &focal_loss_softmax_backward_cuda,
         "softmax_backward_cuda backward (CUDA)");

  // pybind vanilla nms
  pybind11::module naive_nms = m.def_submodule("naive_nms",
                                               "vanilla nms method");
  naive_nms.def("gpu_nms", &gpu_nms, "gpu_nms (CUDA)");
  naive_nms.def("cpu_nms", &cpu_nms, "cpu_nms (CPU)");
  
  // pybind ROIPooling
  pybind11::module rp = m.def_submodule("roi_pooling",
                                        "roi pooling method");
  rp.def("forward_cuda", &roi_pooling_forward_cuda, "roi_pooling forward (CUDA)");
  rp.def("backward_cuda", &roi_pooling_backward_cuda, "roi_pooling backward (CUDA)");
  rp.def("forward_cpu", &roi_pooling_forward, "roi_pooling forward (CPU)");
  
  // pybind ROIAlignPooling
  pybind11::module ra = m.def_submodule("roi_align",
                                        "roi alignment pooling");
  ra.def("forward_cuda", &roi_align_forward_cuda, "roi_align forward (CUDA)");
  ra.def("backward_cuda", &roi_align_backward_cuda, "roi_align backward (CUDA)");
  ra.def("forward_cpu", &roi_align_forward, "roi_align forward (CPU)");
  
  // pybind PSROIPooling
  pybind11::module pp = m.def_submodule("psroi_pooling",
                                        "position sensetive roi pooling");
  pp.def("forward_cuda", &psroi_pooling_forward_cuda, "psroi_pooling forward (CUDA)");
  pp.def("backward_cuda", &psroi_pooling_backward_cuda, "psroi_pooling backward (CUDA)");
  pp.def("forward_cpu", &psroi_pooling_forward, "psroi_pooling forward (CPU)");

  // pybind PSROIMaskPooling layer
  pybind11::module pmp = m.def_submodule("psroi_mask_pooling",
                                         "position sensetive roi pooling with more flexible option");
  pmp.def("forward_cuda", &psroi_mask_pooling_forward_cuda, "psroi mask forward (CUDA)");
  pmp.def("backward_cuda", &psroi_mask_pooling_backward_cuda, "psroi mask backward (CUDA)");
  pmp.def("forward_cpu", &psroi_mask_pooling_forward, "psroi mask forward (CPU)");

  // pybind IOUOverlap
  pybind11::module iou = m.def_submodule("overlaps",
                                         "calculate iou between bboxes & gts");
  iou.def("iou", &gpu_iou_overlaps, "bbox iou overlaps with gt (CUDA)");
}
