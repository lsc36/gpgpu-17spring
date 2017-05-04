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

void CountPosition2(const char *text, int *pos, int text_size)
{
}
