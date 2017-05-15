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

__global__ void CalcAB(const float *background, const float *target,
                       const float *mask, unsigned *A, float *B,
                       const int wb, const int hb,
                       const int wt, const int ht,
                       const int oy, const int ox)
{
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
    const int xt = blockIdx.x * blockDim.x + threadIdx.x;
    const int yb = oy + yt, xb = ox + xt;
    const int curt = wt * yt + xt, curb = wb * yb + xb;
    if (!(yt < ht && xt < wt && mask[curt] > 127.0 &&
          0 <= yb && yb < hb && 0 <= xb && xb < wb))
        return;

    // A: 6543210
    //    DURLNNN
    int a = 0;
    float b0 = 0.0, b1 = 0.0, b2 = 0.0;

#define DoNeighbor1(condb, condt, diffb, difft, b_off)                      \
    if (condb) {                                                            \
        a++;                                                                \
        if (condt) {                                                        \
            b0 += target[curt * 3 + 0] - target[(curt + difft) * 3 + 0];    \
            b1 += target[curt * 3 + 1] - target[(curt + difft) * 3 + 1];    \
            b2 += target[curt * 3 + 2] - target[(curt + difft) * 3 + 2];    \
            if (mask[curt + difft] < 128.0) {                               \
                b0 += background[(curb + diffb) * 3 + 0];                   \
                b1 += background[(curb + diffb) * 3 + 1];                   \
                b2 += background[(curb + diffb) * 3 + 2];                   \
            } else {                                                        \
                a |= (1 << b_off);                                          \
            }                                                               \
        } else {                                                            \
            b0 += background[(curb + diffb) * 3 + 0];                       \
            b1 += background[(curb + diffb) * 3 + 1];                       \
            b2 += background[(curb + diffb) * 3 + 2];                       \
        }                                                                   \
    }                                                                       \

    DoNeighbor1(xb > 0     , xt > 0     ,  -1,  -1, 3);
    DoNeighbor1(xb < wb - 1, xt < wt - 1,   1,   1, 4);
    DoNeighbor1(yb > 0     , yt > 0     , -wb, -wt, 5);
    DoNeighbor1(yb < hb - 1, yt < ht - 1,  wb,  wt, 6);

    A[curt] = a;
    B[curt] = b0;
    B[curt + wt * ht] = b1;
    B[curt + wt * ht * 2] = b2;
}

__global__ void JacobiIteration(const unsigned *A, const float *B,
                                const float *mask, float *output,
                                const int wb, const int hb,
                                const int wt, const int ht,
                                const int oy, const int ox)
{
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
    const int xt = blockIdx.x * blockDim.x + threadIdx.x;
    const int yb = oy + yt, xb = ox + xt;
    const int curt = wt * yt + xt, curb = wb * yb + xb;
    if (!(yt < ht && xt < wt && mask[curt] > 127.0 &&
          0 <= yb && yb < hb && 0 <= xb && xb < wb))
        return;

    unsigned a = A[curt];
    float ax = 0.0;

    if (a & (1 << 3)) ax -= output[curb - 1];
    if (a & (1 << 4)) ax -= output[curb + 1];
    if (a & (1 << 5)) ax -= output[curb - wb];
    if (a & (1 << 6)) ax -= output[curb + wb];

    output[curb] = (B[curt] - ax) / (a & 0x7);
}

template <typename T>
__global__ void Reshape1(const T *input, T *output, const int w, const int h)
{
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(y < h && x < w))
        return;

    const int cur = w * y + x;
    output[cur] = input[cur * 3];
    output[cur + w * h] = input[cur * 3 + 1];
    output[cur + w * h * 2] = input[cur * 3 + 2];
}

template <typename T>
__global__ void Reshape2(const T *input, T *output, const int w, const int h)
{
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(y < h && x < w))
        return;

    const int cur = w * y + x;
    output[cur * 3] = input[cur];
    output[cur * 3 + 1] = input[cur + w * h];
    output[cur * 3 + 2] = input[cur + w * h * 2];
}

const dim3 BS(32, 16);
const int ITER = 20000;

void PoissonImageCloning(const float *background, const float *target,
                         const float *mask, float *output,
                         const int wb, const int hb,
                         const int wt, const int ht,
                         const int oy, const int ox)
{
    unsigned *A = nullptr;
    float *B = nullptr, *buf = nullptr;
    cudaMalloc(&A, wt * ht * sizeof(unsigned));
    cudaMalloc(&B, wt * ht * sizeof(float) * 3);
    cudaMalloc(&buf, wb * hb * sizeof(float) * 3);

    CalcAB<<<dim3(CeilDiv(wt, BS.x), CeilDiv(ht, BS.y)), BS>>>(
        background, target, mask, A, B,
        wb, hb, wt, ht, oy, ox
    );

    Reshape1<<<dim3(CeilDiv(wb, BS.x), CeilDiv(hb, BS.y)), BS>>>(
        background, buf, wb, hb
    );

    for (int i = 0; i < ITER; i++) {
        cudaDeviceSynchronize();
        JacobiIteration<<<dim3(CeilDiv(wt, BS.x), CeilDiv(ht, BS.y)), BS>>>(
            A, B, mask, buf,
            wb, hb, wt, ht, oy, ox
        );
        JacobiIteration<<<dim3(CeilDiv(wt, BS.x), CeilDiv(ht, BS.y)), BS>>>(
            A, B + wt * ht, mask, buf + wb * hb,
            wb, hb, wt, ht, oy, ox
        );
        JacobiIteration<<<dim3(CeilDiv(wt, BS.x), CeilDiv(ht, BS.y)), BS>>>(
            A, B + wt * ht * 2, mask, buf + wb * hb * 2,
            wb, hb, wt, ht, oy, ox
        );
    }

    cudaDeviceSynchronize();
    Reshape2<<<dim3(CeilDiv(wb, BS.x), CeilDiv(hb, BS.y)), BS>>>(
        buf, output, wb, hb
    );

    cudaFree(A);
    cudaFree(B);
    cudaFree(buf);
}
