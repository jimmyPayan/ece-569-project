// Ryan Raad 2025
// normalizeAndTruncate.cu
// most naive Cuda implementation, single pass, single kernel, worst speedup
#include "normalizeAndTruncate.cuh"
#include <cuda_runtime.h>
#include <cmath>

__global__ void normalizeAndTruncateNaive(
    const float* __restrict__ mapData,
    float*       __restrict__ outData,
    int oldSizeX, int oldSizeY,
    float alfa)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int newSizeX = oldSizeX - 2;
    int newSizeY = oldSizeY - 2;
    int cells = newSizeX * newSizeY;
    const int p  = NUM_SECTOR;
    const int xp = NUM_SECTOR*3;
    const int pp = NUM_SECTOR*12;

    if (idx < cells) {
        int j = (idx % newSizeX) + 1;   // 1…oldSizeX–2
        int i = (idx / newSizeX) + 1;   // 1…oldSizeY–2

        // 1) on-the-fly compute the norm of the 4 cells
        float sum0=0, sum1=0, sum2=0, sum3=0;
        int offsets[4] = {
          ( i   * oldSizeX +  j   )*xp,
          ( i   * oldSizeX + (j+1))*xp,
          ((i+1)* oldSizeX +  j   )*xp,
          ((i+1)* oldSizeX + (j+1))*xp
        };
        for (int c=0; c<p; ++c){
          float v;
          v = mapData[offsets[0]+c]; sum0 += v*v;
          v = mapData[offsets[1]+c]; sum1 += v*v;
          v = mapData[offsets[2]+c]; sum2 += v*v;
          v = mapData[offsets[3]+c]; sum3 += v*v;
        }
        float norm = sqrtf(sum0+sum1+sum2+sum3 + 1e-6f);

        // 2) normalize & clamp each of the pp features
        int outBase = idx * pp;
        for (int c=0; c<pp; ++c) {
          // pick corresponding mapData element (this is simplified—your real layout may vary)
          float val = mapData[offsets[0] + c];  
          val = val / norm;
          outData[outBase + c] = (val > alfa ? alfa : val);
        }
    }
}

// Host wrapper omitted for brevity—just cudaMalloc/copy, launch this kernel, copy back.
