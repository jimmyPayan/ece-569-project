#define NUM_SECTOR 9
#define BLOCK_X 6
#define BLOCK_Y 6
#define BLOCK_Z 27
#define K_MAX 4
#define FEATURES_MAX 27

#include <stdio.h>

// Timing functions give tools to directly evaluate speedups! Comment out if you want max performance, I suppose.

/*  NOTE: Clock Rate = 1328500 kHz */

/* 	UPDATE THIS COMMENT AFTER ANY AND ALL OPTIMIZATIONS PLEASE <3 
	As a convention, just use the last thread measurement and copy alongside the Function Timing Summary
		

	OPTIMIZATION NAME: 6x6x27 Fully Active Threads with Privatization and optimized reads

	SINGLESCALE
	Thread 0,0,0 of Block 0,0 took 23402 total cycles. It required:
	~ 1139 cycles to write to shared memory.
	~ 21108 cycles to compute data.
	~ 431 cycles to convert shared memory to global memory.
	724 cycles unaccounted for.
	Total execution time: 96370.0 ms.

	--- Function Timing Summary ---
	Total time spent in getFeatures():24.5 s
	Total time spent in gaussianCorrelation(): 63.5 s
	Total time spent in train(): 7.9 s
	Total time spent in detect(): 8.0 s
	Total time spent in getFeatureMaps():16.8 s

*/

__global__ void kernel_n4(int sizeY, int sizeX, int k, int height, int width, int numFeatures,
                          float *d_map, int stringSize, int *d_alfa, float *d_r, float *d_w, int *d_nearest)
{
    long long int phase0 = clock64();

    __shared__ float shared_w[K_MAX * 2];
    __shared__ int shared_nearest[K_MAX];

    int i = blockIdx.y * BLOCK_Y + threadIdx.y;
    int j = blockIdx.x * BLOCK_X + threadIdx.x;
    int f = threadIdx.z;  // feature index (0..26)

    int phase1, phase2, phase3;

    int d;
    int nearest_ii, nearest_jj;
    int d_alfa_0, d_alfa_1;
    float w_ii_0, w_ii_1, w_jj_0, w_jj_1;
    float d_r_d;
    float acc = 0.0f;

    // Phase 1 Start: Initialize shared memory
    phase1 = clock64();
/* giving errors
    if (threadIdx.y == 0 && threadIdx.z == 0 && threadIdx.x < k * 2) {
        shared_w[threadIdx.x] = d_w[threadIdx.x];
    }
    if (threadIdx.y == 0 && threadIdx.z == 0 && threadIdx.x < k) {
        shared_nearest[threadIdx.x] = d_nearest[threadIdx.x];
    }
*/

if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for (int t = 0; t < k * 2; ++t) {
        shared_w[t] = d_w[t];
    }
    for (int t = 0; t < k; ++t) {
        shared_nearest[t] = d_nearest[t];
    }
}

//__syncthreads();
    __syncthreads();
    phase1 = (int)(clock64() - phase1);

    // Phase 2 Start: Compute
    phase2 = clock64();

    if (i < sizeY && j < sizeX && f < numFeatures) {
        for (int ii = 0; ii < k; ii++) {
            nearest_ii = shared_nearest[ii];
            w_ii_0 = shared_w[ii * 2];
            w_ii_1 = shared_w[ii * 2 + 1];

            for (int jj = 0; jj < k; jj++) {
                int y = i * k + ii;
                int x = j * k + jj;

                if (y > 0 && y < height - 1 && x > 0 && x < width - 1) {
                    d = y * width + x;

                    d_alfa_0 = d_alfa[d * 2];
                    d_alfa_1 = d_alfa[d * 2 + 1];

                    if (d_alfa_0 >= numFeatures || d_alfa_1 + NUM_SECTOR >= numFeatures) continue;  //FIX

                    nearest_jj = shared_nearest[jj];
                    w_jj_0 = shared_w[jj * 2];
                    w_jj_1 = shared_w[jj * 2 + 1];
                    d_r_d = d_r[d];

                    // Center
                    if (f == d_alfa_0)
                        acc += d_r_d * w_ii_0 * w_jj_0;
                    if (f == d_alfa_1 + NUM_SECTOR)
                        acc += d_r_d * w_ii_0 * w_jj_0;

                    // Neighbor in Y
                    int ni = i + nearest_ii;
                    if (ni >= 0 && ni < sizeY) {
                        if (f == d_alfa_0)
                            acc += d_r_d * w_ii_1 * w_jj_0;
                        if (f == d_alfa_1 + NUM_SECTOR)
                            acc += d_r_d * w_ii_1 * w_jj_0;
                    }

                    // Neighbor in X
                    int nj = j + nearest_jj;
                    if (nj >= 0 && nj < sizeX) {
                        if (f == d_alfa_0)
                            acc += d_r_d * w_ii_0 * w_jj_1;
                        if (f == d_alfa_1 + NUM_SECTOR)
                            acc += d_r_d * w_ii_0 * w_jj_1;
                    }

                    // Neighbor in XY
                    if (ni >= 0 && ni < sizeY && nj >= 0 && nj < sizeX) {
                        if (f == d_alfa_0)
                            acc += d_r_d * w_ii_1 * w_jj_1;
                        if (f == d_alfa_1 + NUM_SECTOR)
                            acc += d_r_d * w_ii_1 * w_jj_1;
                    }
                }
            }
        }
    }

    __syncthreads();
    phase2 = (int)(clock64() - phase2);

    // Phase 3 Start: Write to global memory
    phase3 = clock64();
    if (i < sizeY && j < sizeX && f < numFeatures) {
        d_map[i * stringSize + j * numFeatures + f] = acc / (float)(k * k);  //FIX normalization
    }
    phase3 = (int)(clock64() - phase3);

    phase0 = clock64() - phase0;
/*
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 &&
        blockIdx.x == 0 && blockIdx.y == 0) {
        printf("Thread 0,0,0 of Block 0,0 took %d total cycles. It required:\n~ %d cycles to write to shared memory.\n~ %d cycles to compute data.\n~ %d cycles to convert shared memory to global memory.\n%d cycles unaccounted for.\n",
            (int)phase0, phase1, phase2, phase3, ((int)phase0 - (phase1 + phase2 + phase3)));
    }
*/
}

void featureGPU(int sizeY, int sizeX, int k, int height, int width, int numFeatures,
                float *map, int stringSize, int *alfa, float *r, float *w, int *nearest)
{
    float *d_map, *d_r, *d_w;
    int *d_alfa, *d_nearest;

    cudaMalloc((void **)&d_map, sizeof(float) * (sizeX * sizeY * numFeatures));
    cudaMalloc((void **)&d_alfa, sizeof(int) * (width * height * 2));
    cudaMalloc((void **)&d_r, sizeof(float) * (width * height));
    cudaMalloc((void **)&d_w, sizeof(float) * (k * 2));
    cudaMalloc((void **)&d_nearest, sizeof(int) * k);

    cudaMemcpy(d_map, map, sizeof(float) * (sizeX * sizeY * numFeatures), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alfa, alfa, sizeof(int) * (width * height * 2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, r, sizeof(float) * (width * height), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, sizeof(float) * (k * 2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nearest, nearest, sizeof(int) * k, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 blocksPerGrid(
        (sizeX + BLOCK_X - 1) / BLOCK_X,
        (sizeY + BLOCK_Y - 1) / BLOCK_Y);

    kernel_n4<<<blocksPerGrid, threadsPerBlock>>>(sizeY, sizeX, k, height, width, numFeatures,
                                                  d_map, stringSize, d_alfa, d_r, d_w, d_nearest);

    cudaDeviceSynchronize(); // Ensure kernel completes before copying back

    cudaMemcpy(map, d_map, sizeof(float) * (sizeX * sizeY * numFeatures), cudaMemcpyDeviceToHost);

    cudaFree(d_map);
    cudaFree(d_alfa);
    cudaFree(d_r);
    cudaFree(d_w);
    cudaFree(d_nearest);
}
