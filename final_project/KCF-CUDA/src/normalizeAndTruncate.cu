// Ryan Raad 2025
// V1 optimization multi kernel
// normalizeAndTruncate.cu
#include "normalizeAndTruncate.cuh"
#include "fhog.hpp"           // CvLSVMFeatureMapCaskade, NUM_SECTOR, LATENT_SVM_OK
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(err) \
  if ((err) != cudaSuccess) { \
    fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  }

// Kernel 1: compute per-cell squared‐norms
__global__ void computePartOfNorm(
    const float* __restrict__ mapData,
    float*       __restrict__ partOfNorm,
    int oldSizeX, int oldSizeY)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = oldSizeX * oldSizeY;
    const int p = NUM_SECTOR;
    if (idx < total) {
        int base = idx * (p * 3);
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < p; ++j) {
            float v = mapData[base + j];
            sum += v * v;
        }
        partOfNorm[idx] = sum;
    }
}

// Kernel 2: normalize & clamp each HOG block
__global__ void normalizeAndClamp(
    const float* __restrict__ mapData,
    const float* __restrict__ partOfNorm,
    float*       __restrict__ newData,
    int oldSizeX, int oldSizeY, float alfa)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int newSizeX = oldSizeX - 2;
    int newSizeY = oldSizeY - 2;
    int cells    = newSizeX * newSizeY;
    const int p  = NUM_SECTOR;
    const int xp = p * 3;
    const int pp = p * 12;
    const float eps = 1e-6f;

    if (idx < cells) {
        int j = (idx % newSizeX) + 1;
        int i = (idx / newSizeX) + 1;

        // gather 4 neighbouring norms
        int off0 = i     * oldSizeX +  j;
        int off1 = i     * oldSizeX + (j+1);
        int off2 = (i+1) * oldSizeX +  j;
        int off3 = (i+1) * oldSizeX + (j+1);
        float n0 = partOfNorm[off0];
        float n1 = partOfNorm[off1];
        float n2 = partOfNorm[off2];
        float n3 = partOfNorm[off3];
        float norm = sqrtf(n0 + n1 + n2 + n3 + eps);

        // pointers into flat arrays
        int inBase  = off0 * xp;
        int outBase = idx * pp;

        // normalize first p channels and next 2p channels
        for (int c = 0; c < p;    ++c) newData[outBase +     c] = mapData[inBase +     c] / norm;
        for (int c = 0; c < 2*p;  ++c) newData[outBase + 4*p + c] = mapData[inBase + p + c] / norm;

        // clamp all pp channels
        for (int c = 0; c < pp;   ++c) {
            float v = newData[outBase + c];
            newData[outBase + c] = (v > alfa ? alfa : v);
        }
    }
}

// Host wrapper
int normalizeAndTruncateGPU(CvLSVMFeatureMapCaskade* map, const float alfa)
{
    int oldSizeX   = map->sizeX;
    int oldSizeY   = map->sizeY;
    const int p    = NUM_SECTOR;
    const int xp   = p * 3;
    const int pp   = p * 12;
    int totalCells = oldSizeX * oldSizeY;
    int newSizeX   = oldSizeX - 2;
    int newSizeY   = oldSizeY - 2;
    int newCells   = newSizeX * newSizeY;

    size_t inBytes  = sizeof(float) * totalCells * xp;
    size_t partBytes= sizeof(float) * totalCells;
    size_t outBytes = sizeof(float) * newCells   * pp;

    // alloc + copy
    float *d_map, *d_part, *d_new;
    CUDA_CHECK(cudaMalloc(&d_map,   inBytes));
    CUDA_CHECK(cudaMemcpy(d_map, map->map, inBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_part, partBytes));
    CUDA_CHECK(cudaMalloc(&d_new,  outBytes));

    // launch computePartOfNorm
    int t1 = 256, b1 = (totalCells + t1 -1)/t1;
    computePartOfNorm<<<b1,t1>>>(d_map, d_part, oldSizeX, oldSizeY);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // launch normalizeAndClamp
    int t2 = 256, b2 = (newCells + t2 -1)/t2;
    normalizeAndClamp<<<b2,t2>>>(d_map, d_part, d_new, oldSizeX, oldSizeY, alfa);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy back
    float* h_new = (float*)malloc(outBytes);
    CUDA_CHECK(cudaMemcpy(h_new, d_new, outBytes, cudaMemcpyDeviceToHost));

    // cleanup GPU
    cudaFree(d_map);
    cudaFree(d_part);
    cudaFree(d_new);

    // update host map
    free(map->map);
    map->map         = h_new;
    map->sizeX       = newSizeX;
    map->sizeY       = newSizeY;
    map->numFeatures = pp;

    return LATENT_SVM_OK;
}
