#define NUM_SECTOR 9
#define BLOCK_DIM 1
#define K_MAX 4
#define FEATURES_MAX 27

#include <stdio.h>

// Timing functions  give tools to directly evaluate speedups! Comment out if you want max performance, I suppose.

/*  NOTE: Clock Rate = 1328500 kHz */

/* 	UPDATE THIS COMMENT AFTER ANY AND ALL OPTIMIZATIONS PLEASE <3 
	As a convention, just use the last thread measurement and copy alongside the Function Timing Summary
		

	OPTIMIZATION NAME: 27x1 Coalesced Memory with minimal global reads.

	MULTISCALE
	Thread 0,0 of Block 0,0 took 6560 total cycles. It required:
	~ 671 cycles to write to shared memory.
	~ 5387 cycles to compute data.
	~ 176 cycles to convert shared memory to global memory.
	326 cycles unaccounted for.
	Total execution time: 18470.0 ms.

	--- Function Timing Summary ---
	Total time spent in getFeatures(): 2.4 s
	Total time spent in gaussianCorrelation(): 0.7 s
	Total time spent in train(): -2.2 s
	Total time spent in detect(): -1.9 s
	Total execution time: -1.0 s


	SINGLESCALE
	Thread 0,0 of Block 0,0 took 10005 total cycles. It required:
	~ 882 cycles to write to shared memory.
	~ 8573 cycles to compute data.
	~ 222 cycles to convert shared memory to global memory.
	328 cycles unaccounted for.
	Total execution time: 103860.0 ms.

	--- Function Timing Summary ---
	Total time spent in getFeatures(): 23.3 s
	Total time spent in gaussianCorrelation(): 74.8 s
	Total time spent in train(): 14.9 s
	Total time spent in detect(): 14.7 s
	Total execution time: 127.7 s



*/


// No removals of if() statements

__global__ void kernel_n4(int sizeY, int sizeX, int k, int height, int width, int numFeatures, float *d_map, int stringSize, int *d_alfa, float *d_r, float *d_w, int *d_nearest) {
	// Phase 0 Start: Overall thread duration
	long long int phase0 = clock64();

	__shared__ float shared_w[K_MAX * 2];
	__shared__ int shared_nearest[K_MAX];
	__shared__ float shared_blockMap[FEATURES_MAX];

	int i = blockIdx.y;
	int j = blockIdx.x;
	int featureIdx = threadIdx.x;  // 0..26

	int phase1, phase2, phase3;

	int d;

	// Local variables for data accessed multiple times
	int nearest_ii, nearest_jj;
	int d_alfa_0, d_alfa_1;
	float w_ii_0, w_ii_1, w_jj_0, w_jj_1;
	float d_r_d;

	// Phase 1 Start: Initialize shared memory
	phase1 = clock64();
	if (featureIdx < k * 2) 
		shared_w[featureIdx] = d_w[featureIdx];

	if (featureIdx < k) 
		shared_nearest[featureIdx] = d_nearest[featureIdx];

	if (featureIdx < numFeatures)
		shared_blockMap[featureIdx] = 0.0f;

	__syncthreads();
	// Phase 1 End: Initialize shared memory
	phase1 = (int)clock64() - phase1;

	// One thread per feature, one block per cell
	// Phase 2 Start: Compute
	phase2 = clock64();
	if (featureIdx == 0) {

		if (i < sizeY && j < sizeX) {
			for (int ii = 0; ii < k; ii++) {
				for (int jj = 0; jj < k; jj++) {
					if ((i * k + ii > 0) && (i * k + ii < height - 1) &&
						(j * k + jj > 0) && (j * k + jj < width  - 1))
					{
						d = (k * i + ii) * width + (j * k + jj);

						nearest_ii = shared_nearest[ii];
						nearest_jj = shared_nearest[jj];
						d_alfa_0 = d_alfa[d * 2];			
						d_alfa_1 = d_alfa[d * 2 + 1];	
						w_ii_0 = shared_w[ii * 2];
						w_ii_1 = shared_w[ii * 2 + 1];
						w_jj_0 = shared_w[jj * 2];
						w_jj_1 = shared_w[jj * 2 + 1];
						d_r_d = d_r[d];

						shared_blockMap[d_alfa_0] += d_r_d * w_ii_0 * w_jj_0;
						shared_blockMap[d_alfa_1 + NUM_SECTOR] += d_r_d * w_ii_0 * w_jj_0;

						if ((i + nearest_ii >= 0) && (i + nearest_ii < sizeY)) {
							shared_blockMap[d_alfa_0] += d_r_d * w_jj_1 * w_jj_0;
							shared_blockMap[d_alfa_1 + NUM_SECTOR] += d_r_d * w_jj_1 * w_jj_0;
						}

						if ((j + nearest_jj >= 0) && (j + nearest_jj < sizeX)) {
							shared_blockMap[d_alfa_0] += d_r_d * w_ii_0 * w_jj_1;
							shared_blockMap[d_alfa_1 + NUM_SECTOR] += d_r_d * w_ii_0 * w_jj_1;
						}

						if ((i + nearest_ii >= 0) && (i + nearest_ii < sizeY) &&
							(j + nearest_jj >= 0) && (j + nearest_jj < sizeX)) {
							shared_blockMap[d_alfa_0] += d_r_d * w_ii_1 * w_jj_1;
							shared_blockMap[d_alfa_1 + NUM_SECTOR] += d_r_d * w_ii_1 * w_jj_1;
						}
					}
				}
			}
		}
	}

	__syncthreads();
	// Phase 2 End: Compute
	phase2 = (int)clock64() - phase2;


	// Phase 3 Start: Write to global memory
	phase3 = clock64();
	if (i < sizeY && j < sizeX && featureIdx < numFeatures) {
		d_map[i * stringSize + j * numFeatures + featureIdx] = shared_blockMap[featureIdx];
	}
	// Phase 3 End: Write to global memory
	phase3 = (int)clock64() - phase3;

	//Phase 0 End: Overall thread duration
	phase0 = clock64() - phase0;

	if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
		printf("Thread 0,0 of Block 0,0 took %d total cycles. It required:\n~ %d cycles to write to shared memory.\n~ %d cycles to compute data.\n~ %d cycles to convert shared memory to global memory.\n%d cycles unaccounted for.\n", (int) phase0, phase1, phase2, phase3, ((int) phase0 - (phase1 + phase2 + phase3)));	
	}
	
}



void featureGPU(int sizeY, int sizeX, int k, int height, int width, int numFeatures, float *map, int stringSize, int *alfa, float *r, float *w, int *nearest){
	/*
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0); // 0 = device ID (first GPU)
	printf("Clock Rate: %d kHz\n", (int)prop.clockRate);
	*/  

	float *d_map, *d_r, *d_w;
    int *d_alfa, *d_nearest;
    cudaMalloc((void**) &d_map, sizeof(float) * (sizeX * sizeY * numFeatures));
    cudaMalloc((void**) &d_alfa, sizeof(int) * (width * height * 2));
    cudaMalloc((void**) &d_r, sizeof(float) * (width * height));
    cudaMalloc((void**) &d_w, sizeof(float) * (k * 2));
    cudaMalloc((void**) &d_nearest, sizeof(int) * k);

    cudaMemcpy(d_map, map, sizeof(float) * (sizeX * sizeY * numFeatures), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alfa, alfa, sizeof(int) * (width * height * 2) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, r, sizeof(float) * (width * height), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_w, w, sizeof(float) * (k * 2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nearest, nearest, sizeof(int) * k, cudaMemcpyHostToDevice); 

    const dim3 threadsPerBlock(FEATURES_MAX, 1);
    const dim3 blocksPerGrid(sizeX, sizeY);

    kernel_n4<<<blocksPerGrid,threadsPerBlock>>>(sizeY, sizeX, k, height, width, numFeatures, d_map, stringSize, d_alfa, d_r, d_w, d_nearest);

    cudaMemcpy(map, d_map, sizeof(float) * (sizeX * sizeY * numFeatures), cudaMemcpyDeviceToHost);

    cudaFree(d_map);
    cudaFree(d_alfa);
    cudaFree(d_r);
    cudaFree(d_w);
    cudaFree(d_nearest);
}

