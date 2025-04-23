#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_SECTOR 9

// Naive 3D Approach first. Using atomics excessively, we might not see speedup at all.  No memory optimizations yet, no removals of if() statements
// k seems to be passed in as cell_size, which is set to 4... for loop should be okay.

__global__ void kernel_n4(int sizeY, int sizeX, int k, int height, int width, int numFeatures, float *d_map, int stringSize, int *d_alfa, float *d_r, float *d_w,  int *d_nearest) {
	
	// Use thread IDs as iterators, same names as joaofaro to keep me sane while debugging
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int ii = blockIdx.z * blockDim.z + threadIdx.z;
	
	int d;
if (i < sizeY && j < sizeX) {
	for(int jj = 0; jj < k; jj++)
          {
            if ((i * k + ii > 0) && 
                (i * k + ii < height - 1) && 
                (j * k + jj > 0) && 
                (j * k + jj < width  - 1))
            {
              d = (k * i + ii) * width + (j * k + jj);
             // d_map[ i * stringSize + j * numFeatures + d_alfa[d * 2    ]] += 
             //    d_r[d] * d_w[ii * 2] * d_w[jj * 2];
	      atomicAdd(&d_map[ i * stringSize + j * numFeatures + d_alfa[d * 2    ]],
                   d_r[d] * d_w[ii * 2] * d_w[jj * 2]);
             
	     // d_map[ i * stringSize + j * numFeatures + d_alfa[d * 2 + 1] + NUM_SECTOR] += 
             //    d_r[d] * d_w[ii * 2] * d_w[jj * 2];
              atomicAdd(&d_map[ i * stringSize + j * numFeatures + d_alfa[d * 2 + 1] + NUM_SECTOR],
	           d_r[d] * d_w[ii * 2] * d_w[jj * 2]);
	      
	      if ((i + d_nearest[ii] >= 0) && 
                  (i + d_nearest[ii] <= sizeY - 1))
              {
             //   d_map[(i + d_nearest[ii]) * stringSize + j * numFeatures + d_alfa[d * 2    ]             ] += 
             //   d_r[d] * d_w[ii * 2 + 1] * d_w[jj * 2 ];
	      atomicAdd(&d_map[(i + d_nearest[ii]) * stringSize + j * numFeatures + d_alfa[d * 2    ],
	          d_r[d] * d_w[ii * 2 + 1] * d_w[jj * 2 ]);

             //   d_map[(i + d_nearest[ii]) * stringSize + j * numFeatures + d_alfa[d * 2 + 1] + NUM_SECTOR] += 
             //     d_r[d] * d_w[ii * 2 + 1] * d_w[jj * 2 ];
	      atomicAdd(&d_map[(i + d_nearest[ii]) * stringSize + j * numFeatures + d_alfa[d * 2 + 1] + NUM_SECTOR],
	            d_r[d] * d_w[ii * 2 + 1] * d_w[jj * 2 ]);

              }
              if ((j + d_nearest[jj] >= 0) && 
                  (j + d_nearest[jj] <= sizeX - 1))
              {
             //   d_map[i * stringSize + (j + d_nearest[jj]) * numFeatures + d_alfa[d * 2    ]             ] += 
             //     d_r[d] * d_w[ii * 2] * d_w[jj * 2 + 1];
	      atomicAdd(&d_map[i * stringSize + (j + d_nearest[jj]) * numFeatures + d_alfa[d * 2    ]             ],
	            d_r[d] * d_w[ii * 2] * d_w[jj * 2 + 1]);          

	     //   d_map[i * stringSize + (j + d_nearest[jj]) * numFeatures + d_alfa[d * 2 + 1] + NUM_SECTOR] += 
             //     d_r[d] * d_w[ii * 2] * d_w[jj * 2 + 1];
	      atomicAdd(&d_map[i * stringSize + (j + d_nearest[jj]) * numFeatures + d_alfa[d * 2 + 1] + NUM_SECTOR],
	            d_r[d] * d_w[ii * 2] * d_w[jj * 2 + 1]);
              }
              if ((i + d_nearest[ii] >= 0) && 
                  (i + d_nearest[ii] <= sizeY - 1) && 
                  (j + d_nearest[jj] >= 0) && 
                  (j + d_nearest[jj] <= sizeX - 1))
              {
             //   d_map[(i + d_nearest[ii]) * stringSize + (j + d_nearest[jj]) * numFeatures + d_alfa[d * 2    ]             ] += 
             //     d_r[d] * d_w[ii * 2 + 1] * d_w[jj * 2 + 1];
	      atomicAdd(&d_map[(i + d_nearest[ii]) * stringSize + (j + d_nearest[jj]) * numFeatures + d_alfa[d * 2    ]],
	            d_r[d] * d_w[ii * 2 + 1] * d_w[jj * 2 + 1]);

             //   d_map[(i + d_nearest[ii]) * stringSize + (j + d_nearest[jj]) * numFeatures + d_alfa[d * 2 + 1] + NUM_SECTOR] += 
             //     d_r[d] * d_w[ii * 2 + 1] * d_w[jj * 2 + 1];
	      atomicAdd(&d_map[(i + d_nearest[ii]) * stringSize + (j + d_nearest[jj]) * numFeatures + d_alfa[d * 2 + 1] + NUM_SECTOR],
	            d_r[d] * d_w[ii * 2 + 1] * d_w[jj * 2 + 1]);
              }
            } // if()
} // for(int jj = 0; jj < k; jj++)
} // if (i < sizeY && j < sizeX)
}
