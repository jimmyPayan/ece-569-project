// Ryan Raad 2025
// normalizeAndTruncate.cu
// most naive Cuda implementation, single pass, single kernel, worst speedup

#include "normalizeAndTruncate.cuh"
#include "fhog.hpp"           // for CvLSVMFeatureMapCaskade, NUM_SECTOR, LATENT_SVM_OK
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>             // for printf
#include <cstdlib>            // for malloc/free

#define CUDA_CHECK(err) \
  if ((err) != cudaSuccess) { \
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  }

// -----------------------------------------------------------------------------
//  Kernel: normalizeAndTruncateNaiveKernel
// -----------------------------------------------------------------------------
__global__ void normalizeAndTruncateNaiveKernel(
    const float* __restrict__ mapData,
    float*       __restrict__ outData,
    int oldSizeX, int oldSizeY,
    float alfa)
{
    int idx       = blockIdx.x*blockDim.x + threadIdx.x;
    int newSizeX  = oldSizeX - 2;
    int newSizeY  = oldSizeY - 2;
    int cells     = newSizeX * newSizeY;
    const int p   = NUM_SECTOR;
    const int xp  = NUM_SECTOR*3;
    const int pp  = NUM_SECTOR*12;

    if (idx < cells) {
        int j = (idx % newSizeX) + 1;   // 1…oldSizeX–2
        int i = (idx / newSizeX) + 1;   // 1…oldSizeY–2

        // compute the four cell‐norms on the fly
        float sum0=0, sum1=0, sum2=0, sum3=0;
        int offsets[4] = {
          ( i   * oldSizeX +  j   )*xp,
          ( i   * oldSizeX + (j+1))*xp,
          ((i+1)* oldSizeX +  j   )*xp,
          ((i+1)* oldSizeX + (j+1))*xp
        };
        for (int c=0; c<p; ++c){
          float v = mapData[offsets[0]+c]; sum0 += v*v;
          v       = mapData[offsets[1]+c]; sum1 += v*v;
          v       = mapData[offsets[2]+c]; sum2 += v*v;
          v       = mapData[offsets[3]+c]; sum3 += v*v;
        }
        float norm = sqrtf(sum0+sum1+sum2+sum3 + 1e-6f);

        // normalize & clamp
        int outBase = idx * pp;
        for (int c=0; c<pp; ++c) {
          float val = mapData[offsets[0] + c] / norm;
          outData[outBase + c] = (val > alfa ? alfa : val);
        }
    }
}

// -----------------------------------------------------------------------------
//  Host wrapper: normalizeAndTruncateNaive()
// -----------------------------------------------------------------------------
int normalizeAndTruncateNaive(CvLSVMFeatureMapCaskade* map, float alfa)
{
    // 1) Gather sizes
    int oldSizeX   = map->sizeX;
    int oldSizeY   = map->sizeY;
    const int p    = NUM_SECTOR;
    const int xp   = p*3;
    const int pp   = p*12;
    int totalCells = oldSizeX * oldSizeY;
    int newSizeX   = oldSizeX - 2;
    int newSizeY   = oldSizeY - 2;
    int newCells   = newSizeX * newSizeY;

    size_t mapBytes = sizeof(float) * totalCells * xp;
    size_t outBytes = sizeof(float) * newCells   * pp;

    // 2) Allocate & copy input map to device
    float *d_map, *d_out;
    CUDA_CHECK(cudaMalloc(&d_map, mapBytes));
    CUDA_CHECK(cudaMemcpy(d_map, map->map, mapBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_out, outBytes));

    // 3) Launch kernel
    int threads = 256;
    int blocks1 = (newCells + threads - 1) / threads;
    normalizeAndTruncateNaiveKernel<<<blocks1, threads>>>(d_map, d_out, oldSizeX, oldSizeY, alfa);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4) Copy result back
    float* h_out = (float*)malloc(outBytes);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, outBytes, cudaMemcpyDeviceToHost));

    // 5) Clean up device
    cudaFree(d_map);
    cudaFree(d_out);

    // 6) Replace host map with new data
    free(map->map);
    map->map         = h_out;
    map->sizeX       = newSizeX;
    map->sizeY       = newSizeY;
    map->numFeatures = pp;

    return LATENT_SVM_OK;
}
