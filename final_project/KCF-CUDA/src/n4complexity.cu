#define NUM_SECTOR 9
#include <stdio.h>

// Naive 3D Approach first. Using atomics excessively, so we might not see speedup at all.  No memory optimizations yet, no removals of if() statements
// k seems to be passed in as cell_size, which is set to 4... for loop should be okay.

__global__ void kernel_n4(int sizeY, int sizeX, int k, int height, int width, int numFeatures, float *d_map, int stringSize, int *d_alfa, float *d_r, float *d_w,  int *d_nearest) {
	
	// Use thread IDs as iterators, same names as joaofaro to keep me sane while debugging
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	//int ii = blockIdx.z * blockDim.z + threadIdx.z;
	
	int d;
if (i < sizeY && j < sizeX) {
	for(int ii = 0; ii < k; ii++) { /////////////////////////////////////
	for(int jj = 0; jj < k; jj++)
          {
            if ((i * k + ii > 0) && 
                (i * k + ii < height - 1) && 
                (j * k + jj > 0) && 
                (j * k + jj < width  - 1))
            {
            d = (k * i + ii) * width + (j * k + jj);
           //  d_map[ i * stringSize + j * numFeatures + d_alfa[d * 2    ]] += 
           //     d_r[d] * d_w[ii * 2] * d_w[jj * 2];
	      atomicAdd(&d_map[ i * stringSize + j * numFeatures + d_alfa[d * 2    ]],
                   d_r[d] * d_w[ii * 2] * d_w[jj * 2]);
             
	   //   d_map[ i * stringSize + j * numFeatures + d_alfa[d * 2 + 1] + NUM_SECTOR] += 
           //      d_r[d] * d_w[ii * 2] * d_w[jj * 2];
              atomicAdd(&d_map[ i * stringSize + j * numFeatures + d_alfa[d * 2 + 1] + NUM_SECTOR],
	           d_r[d] * d_w[ii * 2] * d_w[jj * 2]);
	      
	      if ((i + d_nearest[ii] >= 0) && 
                  (i + d_nearest[ii] <= sizeY - 1))
              {
           //     d_map[(i + d_nearest[ii]) * stringSize + j * numFeatures + d_alfa[d * 2    ]             ] += 
           //     d_r[d] * d_w[ii * 2 + 1] * d_w[jj * 2 ];
	      atomicAdd(&d_map[(i + d_nearest[ii]) * stringSize + j * numFeatures + d_alfa[d * 2    ]],
	          d_r[d] * d_w[ii * 2 + 1] * d_w[jj * 2 ]);

           //     d_map[(i + d_nearest[ii]) * stringSize + j * numFeatures + d_alfa[d * 2 + 1] + NUM_SECTOR] += 
           //       d_r[d] * d_w[ii * 2 + 1] * d_w[jj * 2 ];
	      atomicAdd(&d_map[(i + d_nearest[ii]) * stringSize + j * numFeatures + d_alfa[d * 2 + 1] + NUM_SECTOR],
	            d_r[d] * d_w[ii * 2 + 1] * d_w[jj * 2 ]);

              }
              if ((j + d_nearest[jj] >= 0) && 
                  (j + d_nearest[jj] <= sizeX - 1))
              {
            //    d_map[i * stringSize + (j + d_nearest[jj]) * numFeatures + d_alfa[d * 2    ]             ] += 
            //      d_r[d] * d_w[ii * 2] * d_w[jj * 2 + 1];
	      atomicAdd(&d_map[i * stringSize + (j + d_nearest[jj]) * numFeatures + d_alfa[d * 2    ]             ],
	            d_r[d] * d_w[ii * 2] * d_w[jj * 2 + 1]);          

	    //    d_map[i * stringSize + (j + d_nearest[jj]) * numFeatures + d_alfa[d * 2 + 1] + NUM_SECTOR] += 
            //      d_r[d] * d_w[ii * 2] * d_w[jj * 2 + 1];
	      atomicAdd(&d_map[i * stringSize + (j + d_nearest[jj]) * numFeatures + d_alfa[d * 2 + 1] + NUM_SECTOR],
	            d_r[d] * d_w[ii * 2] * d_w[jj * 2 + 1]);
              }
              if ((i + d_nearest[ii] >= 0) && 
                  (i + d_nearest[ii] <= sizeY - 1) && 
                  (j + d_nearest[jj] >= 0) && 
                  (j + d_nearest[jj] <= sizeX - 1))
              {
            //    d_map[(i + d_nearest[ii]) * stringSize + (j + d_nearest[jj]) * numFeatures + d_alfa[d * 2    ]             ] += 
            //      d_r[d] * d_w[ii * 2 + 1] * d_w[jj * 2 + 1];
	      atomicAdd(&d_map[(i + d_nearest[ii]) * stringSize + (j + d_nearest[jj]) * numFeatures + d_alfa[d * 2    ]],
	            d_r[d] * d_w[ii * 2 + 1] * d_w[jj * 2 + 1]);

            //    d_map[(i + d_nearest[ii]) * stringSize + (j + d_nearest[jj]) * numFeatures + d_alfa[d * 2 + 1] + NUM_SECTOR] += 
            //      d_r[d] * d_w[ii * 2 + 1] * d_w[jj * 2 + 1];
	      atomicAdd(&d_map[(i + d_nearest[ii]) * stringSize + (j + d_nearest[jj]) * numFeatures + d_alfa[d * 2 + 1] + NUM_SECTOR],
	            d_r[d] * d_w[ii * 2 + 1] * d_w[jj * 2 + 1]);
              }
            } // if()
} // for(int jj = 0; jj < k; jj++)
} // for(int ii = 0; ii < k; ii++)
} // if (i < sizeY && j < sizeX)
}


void featureGPU(int sizeY, int sizeX, int k, int height, int width, int numFeatures, float *map, int stringSize, int *alfa, float *r, float *w, int *nearest){
	printf("Running kernel_n4.\n");
    float *d_map, *d_r, *d_w;
    int *d_alfa, *d_nearest;
// Commented out cudaMalloc()'s are passed by value
    //cudaMalloc((void**) &sizeY, sizeof(int));
    //cudaMalloc((void**) &sizeX, sizeof(int));
    //cudaMalloc((void**) &k,     sizeof(int));
    //cudaMalloc((void**) &height, sizeof(int);
    //cudaMalloc((void**) &width, sizeof(int));
    //cudaMalloc((void**) &numFeatures, sizeof(int));
    cudaMalloc((void**) &d_map, sizeof(float) * (sizeX * sizeY * numFeatures));
    //cudaMalloc((void**) &stringSize, sizeof(int));
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
    const dim3 threadsPerBlock(32,32);
    const dim3 blocksPerGrid(ceil((float)sizeY / 32), ceil((float)sizeX / 32));

    kernel_n4<<<blocksPerGrid,threadsPerBlock>>>(sizeY, sizeX, k, height, width, numFeatures, map, stringSize, alfa, r, w, nearest);

    cudaMemcpy(map, d_map, sizeof(float) * (sizeX * sizeY * numFeatures), cudaMemcpyDeviceToHost);
    cudaMemcpy(alfa, d_alfa, sizeof(int) * (width * height * 2) , cudaMemcpyDeviceToHost);
    cudaMemcpy(r, d_r, sizeof(float) * (width * height), cudaMemcpyDeviceToHost); 
    cudaMemcpy(w, d_w, sizeof(float) * (k * 2), cudaMemcpyDeviceToHost);
    cudaMemcpy(nearest, d_nearest, sizeof(int) * k, cudaMemcpyDeviceToHost); 

    cudaFree(d_map);
    cudaFree(d_alfa);
    cudaFree(d_r);
    cudaFree(d_w);
    cudaFree(d_nearest);
}
