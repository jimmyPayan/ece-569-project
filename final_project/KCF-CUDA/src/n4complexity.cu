#define NUM_SECTOR 9
#define BLOCK_DIM 16
#define K_MAX 4
#define FEATURES_MAX 27

#include <stdio.h>

// First optimization: Proper block dimensions (32x32 -> 16x16). Preparatory optimization that resulted in -10s from singlescale and -2s from Multiscale. Done so that shared memory (optimization 2) will not be > 48 kB.

// Still using atomics excessively, so we might not see speedup at all.  No removals of if() statements
// k seems to be passed in as cell_size, which is set to 4... for loop should be okay.

__global__ void kernel_n4(int sizeY, int sizeX, int k, int height, int width, int numFeatures, float *d_map, int stringSize, int *d_alfa, float *d_r, float *d_w,  int *d_nearest) {
	
	// Allocate shared memory
	__shared__ float shared_w[K_MAX * 2];
	__shared__ int shared_nearest[K_MAX];
	__shared__ float shared_blockMap[BLOCK_DIM * BLOCK_DIM * FEATURES_MAX];

	// Use thread IDs as iterators, same names as joaofaro to keep me sane while debugging
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int Idx = threadIdx.x + threadIdx.y * blockDim.x;	

	int a, d;

	int nearest_ii, nearest_jj;

	if (Idx < k * 2)
		shared_w [Idx] = d_w[Idx];

	if (Idx < k)
		shared_nearest[Idx] = d_nearest[Idx];

	if (Idx < (BLOCK_DIM * BLOCK_DIM)) {
		for (a = 0 ; a < numFeatures; a++) {
			shared_blockMap[Idx * numFeatures + a] = 0.0f;
		}
	}

	__syncthreads();

if (i < sizeY && j < sizeX) {
    for (int ii = 0; ii < k; ii++) {
    for (int jj = 0; jj < k; jj++) {
        if ((i * k + ii > 0) && 
            (i * k + ii < height - 1) && 
            (j * k + jj > 0) && 
            (j * k + jj < width - 1))
        {
            d = (k * i + ii) * width + (j * k + jj);
			
			nearest_ii = shared_nearest[ii];
			nearest_jj = shared_nearest[jj];

            shared_blockMap[Idx * numFeatures + d_alfa[d * 2]] += d_r[d] * shared_w[ii * 2] * shared_w[jj * 2];

            shared_blockMap[Idx * numFeatures + d_alfa[d * 2 + 1] + NUM_SECTOR] += d_r[d] * shared_w[ii * 2] * shared_w[jj * 2];

			
            if (((int)threadIdx.x + nearest_ii >= 0) && 
                ((int)threadIdx.x + nearest_ii < BLOCK_DIM))
            {

                shared_blockMap[((threadIdx.x + nearest_ii) + threadIdx.y * blockDim.x) * numFeatures + d_alfa[d * 2]] += d_r[d] * shared_w[ii * 2 + 1] * shared_w[jj * 2];

                shared_blockMap[((threadIdx.x + nearest_ii) + threadIdx.y * blockDim.x) * numFeatures + d_alfa[d * 2 + 1] + NUM_SECTOR] += d_r[d] * shared_w[ii * 2 + 1] * shared_w[jj * 2];
            }
			
			
            if (((int)threadIdx.y + nearest_jj >= 0) && 
                ((int)threadIdx.y + nearest_jj < BLOCK_DIM))
            {

                shared_blockMap[(threadIdx.x + (threadIdx.y + shared_nearest[jj]) * blockDim.x) * numFeatures + d_alfa[d * 2]] += d_r[d] * shared_w[ii * 2] * shared_w[jj * 2 + 1];

                shared_blockMap[(threadIdx.x + (threadIdx.y + shared_nearest[jj]) * blockDim.x) * numFeatures + d_alfa[d * 2 + 1] + NUM_SECTOR] += d_r[d] * shared_w[ii * 2] * shared_w[jj * 2 + 1];
            }

            if (((int)threadIdx.x + nearest_ii 	>= 0) && 
                ((int)threadIdx.x + nearest_ii 	< BLOCK_DIM) &&
                ((int)threadIdx.y + nearest_jj  >= 0) && 
                ((int)threadIdx.y + nearest_jj	< BLOCK_DIM))
            {

                shared_blockMap[((threadIdx.x + nearest_ii) + (threadIdx.y + nearest_jj) * blockDim.x) * numFeatures + d_alfa[d * 2]] += d_r[d] * shared_w[ii * 2 + 1] * shared_w[jj * 2 + 1];

                shared_blockMap[((threadIdx.x + nearest_ii) + (threadIdx.y + nearest_jj) * blockDim.x) * numFeatures + d_alfa[d * 2 + 1] + NUM_SECTOR] += d_r[d] * shared_w[ii * 2 + 1] * shared_w[jj * 2 + 1];
            }
        }
    }/*for(jj = 0; jj < k; jj++)*/
    }/*for(ii = 0; ii < k; ii++)*/
}/*if (i < sizeY && j < sizeX)*/

__syncthreads();

// Write to global memory
for (int a = 0; a < numFeatures; a++) {
    d_map[i * stringSize + j * numFeatures + a] = shared_blockMap[Idx * numFeatures + a];
}

}

void featureGPU(int sizeY, int sizeX, int k, int height, int width, int numFeatures, float *map, int stringSize, int *alfa, float *r, float *w, int *nearest){
	//printf("Running kernel_n4.\n");
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

    // Total number of threads needed: sizeY * sizeX * k... max value of k is 4, which occurs during HOG. 1024 / 4 = 256, sqrt(256) = 16.
    const dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM);
    const dim3 blocksPerGrid(ceil((float)sizeY / BLOCK_DIM), ceil((float)sizeX / BLOCK_DIM));

    kernel_n4<<<blocksPerGrid,threadsPerBlock>>>(sizeY, sizeX, k, height, width, numFeatures, d_map, stringSize, d_alfa, d_r, d_w, d_nearest);

    cudaMemcpy(map, d_map, sizeof(float) * (sizeX * sizeY * numFeatures), cudaMemcpyDeviceToHost);

    cudaFree(d_map);
    cudaFree(d_alfa);
    cudaFree(d_r);
    cudaFree(d_w);
    cudaFree(d_nearest);
}
