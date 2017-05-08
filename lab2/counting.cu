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

__device__ void MySegmentedScanInit(const char *text, int *pos, int text_size)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x, tid = threadIdx.x;

    if (i < text_size)
        pos[tid] = text[i] == '\n' ? 0 : 1;
    else
        pos[tid] = 0;

    __syncthreads();
}

__device__ void MySegmentedScanIntraWarp(int *pos)
{
    const int lane = threadIdx.x & 31;

    if (lane >=  1) pos[lane] += pos[lane -  1] * (pos[lane] & 1);
    if (lane >=  2) pos[lane] += pos[lane -  2] * ((pos[lane] >> 1) & 1);
    if (lane >=  4) pos[lane] += pos[lane -  4] * ((pos[lane] >> 2) & 1);
    if (lane >=  8) pos[lane] += pos[lane -  8] * ((pos[lane] >> 3) & 1);
    if (lane >= 16) pos[lane] += pos[lane - 16] * ((pos[lane] >> 4) & 1);
}

__device__ void MySegmentedScanMergeWarps(int *pos, int level)
{
    const int lane = threadIdx.x & 31;
    int tmp = pos[lane];

    if (tmp == (32 << level) - 32 + lane + 1)
        tmp += pos[-(32 << level) + 31];
    __syncthreads();

    pos[lane] = tmp;
}

__device__ void MySegmentedScanIntraBlock(int *pos)
{
    const int wid = threadIdx.x >> 5;
    const int offset = wid << 5;

    MySegmentedScanIntraWarp(pos + offset);
    __syncthreads();

    if (wid >= 1) MySegmentedScanMergeWarps(pos + offset, 0);
    __syncthreads();
    if (wid >= 2) MySegmentedScanMergeWarps(pos + offset, 1);
    __syncthreads();
    if (wid >= 4) MySegmentedScanMergeWarps(pos + offset, 2);
    __syncthreads();
    if (wid >= 8) MySegmentedScanMergeWarps(pos + offset, 3);
    __syncthreads();
}

__device__ void MySegmentedScanMergeBlocks(int *pos)
{
    const int tid = threadIdx.x;
    int tmp = pos[tid];

    if (tmp == tid + 1) tmp += pos[-1];

    pos[tid] = tmp;
}

__global__ void MySegmentedScanStep1(const char *text, int *pos, int text_size)
{
    const int bid = blockIdx.x, tid = threadIdx.x;
    __shared__ int pos2[512];

    MySegmentedScanInit(text, pos2, text_size);
    MySegmentedScanIntraBlock(pos2);

    const int i = 512 * bid + tid;
    if (i < text_size) pos[i] = pos2[tid];
}

__global__ void MySegmentedScanStep2(int *pos)
{
    const int bid = blockIdx.x;

    MySegmentedScanMergeBlocks(pos + 512 * (bid + 1));
}

void CountPosition2(const char *text, int *pos, int text_size)
{
    const int size = CeilAlign(text_size, 512);

    MySegmentedScanStep1<<<size / 512, 512, 512 * sizeof(int)>>>(text, pos,
                                                                 text_size);
    cudaDeviceSynchronize();
    MySegmentedScanStep2<<<size / 512 - 1, 512>>>(pos);
}
