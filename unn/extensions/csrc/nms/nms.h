#ifndef NMS_H_
#define NMS_H_

#include <ATen/ATen.h>
#include <cmath>
#include <cstdio>
#include <cfloat>

int gpu_nms(at::Tensor keep, at::Tensor num_out, at::Tensor boxes, float nms_overlap_thresh);
int cpu_nms(at::Tensor keep_out, at::Tensor num_out, at::Tensor boxes, at::Tensor order,
            at::Tensor areas, float nms_overlap_thresh);

#endif
