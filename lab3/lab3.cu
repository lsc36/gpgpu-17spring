#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(const float *background, const float *target,
                            const float *mask, float *output,
                            const int wb, const int hb,
                            const int wt, const int ht,
                            const int oy, const int ox)
{
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
    const int xt = blockIdx.x * blockDim.x + threadIdx.x;
    const int curt = wt * yt + xt;
    if (yt < ht && xt < wt && mask[curt] > 127.0f) {
        const int yb = oy + yt, xb = ox + xt;
        const int curb = wb * yb + xb;
        if (0 <= yb && yb < hb && 0 <= xb && xb < wb) {
            output[curb * 3 + 0] = target[curt * 3 + 0];
            output[curb * 3 + 1] = target[curt * 3 + 1];
            output[curb * 3 + 2] = target[curt * 3 + 2];
        }
    }
}

const dim3 BS(32, 16);

void PoissonImageCloning(const float *background, const float *target,
                         const float *mask, float *output,
                         const int wb, const int hb,
                         const int wt, const int ht,
                         const int oy, const int ox)
{
    cudaMemcpy(output, background, wb * hb * sizeof(float) * 3,
               cudaMemcpyDeviceToDevice);
    SimpleClone<<<dim3(CeilDiv(wt, BS.x), CeilDiv(ht, BS.y)), BS>>>(
        background, target, mask, output,
        wb, hb, wt, ht, oy, ox
    );
}
