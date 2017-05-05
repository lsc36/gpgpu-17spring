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
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < text_size)
        pos[i] = text[i] == '\n' ? 0 : 1;
    else
        pos[i] = 0;

    __syncthreads();
}

__device__ void MySegmentedScanIntraBlock(int *pos)
{
    const int tid = threadIdx.x;
    int tmp;

    if (tid >=   1) {
	tmp = pos[tid -   1]; __syncthreads();
	if (pos[tid] ==   1) pos[tid] += tmp;
    }
    __syncthreads();
    if (tid >=   2) {
	tmp = pos[tid -   2]; __syncthreads();
	if (pos[tid] ==   2) pos[tid] += tmp;
    }
    __syncthreads();
    if (tid >=   4) {
	tmp = pos[tid -   4]; __syncthreads();
	if (pos[tid] ==   4) pos[tid] += tmp;
    }
    __syncthreads();
    if (tid >=   8) {
	tmp = pos[tid -   8]; __syncthreads();
	if (pos[tid] ==   8) pos[tid] += tmp;
    }
    __syncthreads();
    if (tid >=  16) {
	tmp = pos[tid -  16]; __syncthreads();
	if (pos[tid] ==  16) pos[tid] += tmp;
    }
    __syncthreads();
    if (tid >=  32) {
	tmp = pos[tid -  32]; __syncthreads();
	if (pos[tid] ==  32) pos[tid] += tmp;
    }
    __syncthreads();
    if (tid >=  64) {
	tmp = pos[tid -  64]; __syncthreads();
	if (pos[tid] ==  64) pos[tid] += tmp;
    }
    __syncthreads();
    if (tid >= 128) {
	tmp = pos[tid - 128]; __syncthreads();
	if (pos[tid] == 128) pos[tid] += tmp;
    }
    __syncthreads();
    if (tid >= 256) {
	tmp = pos[tid - 256]; __syncthreads();
	if (pos[tid] == 256) pos[tid] += tmp;
    }
    __syncthreads();
}

__device__ void MySegmentedScanMergeBlocks(int *pos)
{
    const int tid = threadIdx.x;

    if (pos[tid] == tid + 1) pos[tid] += pos[-1];
}

__global__ void MySegmentedScanStep1(const char *text, int *pos, int text_size)
{
    const int bid = blockIdx.x;

    MySegmentedScanInit(text, pos, text_size);
    MySegmentedScanIntraBlock(pos + 512 * bid);
}

__global__ void MySegmentedScanStep2(int *pos)
{
    const int bid = blockIdx.x;

    MySegmentedScanMergeBlocks(pos + 512 * (bid + 1));
}

void CountPosition2(const char *text, int *pos, int text_size)
{
    int *pos2 = nullptr;
    const int size = CeilAlign(text_size, 512);
    cudaMalloc(&pos2, size * sizeof(int));

    MySegmentedScanStep1<<<size / 512, 512>>>(text, pos2, text_size);
    cudaDeviceSynchronize();
    MySegmentedScanStep2<<<size / 512 - 1, 512>>>(pos2);

    cudaMemcpy(pos, pos2, text_size * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaFree(pos2);
}
