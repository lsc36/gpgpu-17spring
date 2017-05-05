#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

struct MyUnaryOp {
    __device__ int operator()(const char &c) const {
        return c == '\n' ? 0 : 1;
    }
};

struct MyBinaryPred {
    __device__ bool operator()(const char &a, const char &b) const {
        return b != '\n';
    }
};

void CountPosition1(const char *text, int *pos, int text_size)
{
    thrust::transform(thrust::system::cuda::par,
                      text, text + text_size, pos, MyUnaryOp());
    thrust::inclusive_scan_by_key(thrust::system::cuda::par,
                                  text, text + text_size, pos, pos,
                                  MyBinaryPred());
}

__global__ void MySegmentedScanInit(const char *text, int *pos1, int *pos2,
                                    int text_size)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < text_size)
        pos1[i] = pos2[i] = text[i] == '\n' ? 0 : 1;
}

__global__ void MySegmentedScanPass(int *pos1, const int *pos2, int text_size,
                                    int offset)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < text_size - offset && pos1[i + offset] == offset)
        pos1[i + offset] += pos2[i];
}

const int BS = 128;

void CountPosition2(const char *text, int *pos, int text_size)
{
    int *pos2 = nullptr;
    cudaMalloc(&pos2, text_size * sizeof(int));

    MySegmentedScanInit<<<CeilDiv(text_size, BS), BS>>>(
            text, pos, pos2, text_size);
    for (int offset = 1; offset < text_size; offset <<= 1) {
        MySegmentedScanPass<<<CeilDiv(text_size, BS), BS>>>(
                pos, pos2, text_size, offset);
        cudaMemcpy(pos2, pos, text_size * sizeof(int),
                   cudaMemcpyDeviceToDevice);
    }

    cudaFree(pos2);
}
