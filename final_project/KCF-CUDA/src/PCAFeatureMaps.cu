#include "PCAFeatureMaps.cuh"
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_CHECK(err) /* same as your other kernels */

// Kernel: one thread per output feature cell+channel
__global__ void pcaKernel(
    const float* __restrict__ inMap,
    float*       __restrict__ outMap,
    int sizeX, int sizeY,
    int inChans, int outChans,
    float normFactor)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int total = sizeX*sizeY*outChans;
  if (idx >= total) return;

  int chan = idx / (sizeX*sizeY);
  int cell = idx % (sizeX*sizeY);
  // sum over inChans:
  float sum = 0.0f;
  for (int k = 0; k < inChans; ++k) {
    sum += inMap[k*(sizeX*sizeY) + cell];
  }
  outMap[chan*(sizeX*sizeY) + cell] = sum * normFactor;
}

int PCAFeatureMapsGPU(CvLSVMFeatureMapCaskade *map) {
  int sizeX   = map->sizeX, sizeY = map->sizeY;
  int inChans = map->numFeatures;      // = pp from normalize step
  int outChans= /* whatever your serial code reduces to, e.g. NUM_SECTOR*4 */;
  int cells   = sizeX*sizeY;
  float normF = /* your serial code’s ny value */;

  size_t inBytes  = sizeof(float)*cells*inChans;
  size_t outBytes = sizeof(float)*cells*outChans;

  // 1) allocate & copy input if you haven’t already
  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in,  inBytes));
  CUDA_CHECK(cudaMalloc(&d_out, outBytes));
  CUDA_CHECK(cudaMemcpy(d_in, map->map, inBytes, cudaMemcpyHostToDevice));

  // 2) kernel launch
  int threads = 256, blocks = (cells*outChans + threads-1)/threads;
  pcaKernel<<<blocks,threads>>>(d_in, d_out, sizeX, sizeY, inChans, outChans, normF);
  CUDA_CHECK(cudaDeviceSynchronize());

  // 3) copy back & clean up
  float *h_out = (float*)malloc(outBytes);
  CUDA_CHECK(cudaMemcpy(h_out, d_out, outBytes, cudaMemcpyDeviceToHost));
  cudaFree(d_in);
  cudaFree(d_out);

  // 4) swap buffers & update map struct
  free(map->map);
  map->map        = h_out;
  map->numFeatures= outChans;
  return LATENT_SVM_OK;
}
