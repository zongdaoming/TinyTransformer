#include "deform_conv/deformable_conv.h"

#include <cstdio>
using at::Tensor;

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int deform_conv_forward(Tensor input, Tensor weight,
                        Tensor offset, Tensor output,
                        Tensor columns, Tensor ones, int kH,
                        int kW, int dH, int dW, int padH, int padW,
                        int dilationH, int dilationW, int groups,
                        int deformable_group) {
    // ONNX requires operations support cpu forward
    return 0;
}
