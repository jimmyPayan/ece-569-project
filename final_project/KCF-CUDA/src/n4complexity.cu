#include "fhog.hpp"


#ifdef HAVE_TBB
#include <tbb/tbb.h>
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#endif

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#define NUM_SECTOR 9

// Naive 3D Approach first. Can do pseudo-4D if needed. No memory optimizations yet, no removals of if() statements
// k seems to be passed in as cell_size, which is set to 4... for loop should be okay.

__globabl__ void kernel_n4(int sizeY, int sizeX, int k, int height, int width, float *map, int stringSize, int *alfa, float *r, float *w,  int *nearest) {
	
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
              map[ i * stringSize + j * (*map)->numFeatures + alfa[d * 2    ]] += 
                  r[d] * w[ii * 2] * w[jj * 2];
              map[ i * stringSize + j * (*map)->numFeatures + alfa[d * 2 + 1] + NUM_SECTOR] += 
                  r[d] * w[ii * 2] * w[jj * 2];
              if ((i + nearest[ii] >= 0) && 
                  (i + nearest[ii] <= sizeY - 1))
              {
                map[(i + nearest[ii]) * stringSize + j * (*map)->numFeatures + alfa[d * 2    ]             ] += 
                  r[d] * w[ii * 2 + 1] * w[jj * 2 ];
                map[(i + nearest[ii]) * stringSize + j * (*map)->numFeatures + alfa[d * 2 + 1] + NUM_SECTOR] += 
                  r[d] * w[ii * 2 + 1] * w[jj * 2 ];
              }
              if ((j + nearest[jj] >= 0) && 
                  (j + nearest[jj] <= sizeX - 1))
              {
                map[i * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[d * 2    ]             ] += 
                  r[d] * w[ii * 2] * w[jj * 2 + 1];
                map[i * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[d * 2 + 1] + NUM_SECTOR] += 
                  r[d] * w[ii * 2] * w[jj * 2 + 1];
              }
              if ((i + nearest[ii] >= 0) && 
                  (i + nearest[ii] <= sizeY - 1) && 
                  (j + nearest[jj] >= 0) && 
                  (j + nearest[jj] <= sizeX - 1))
              {
                map[(i + nearest[ii]) * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[d * 2    ]             ] += 
                  r[d] * w[ii * 2 + 1] * w[jj * 2 + 1];
                map[(i + nearest[ii]) * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[d * 2 + 1] + NUM_SECTOR] += 
                  r[d] * w[ii * 2 + 1] * w[jj * 2 + 1];
              }
            } // for(int jj = 0; jj < k; jj++)
} // if (i < sizeY && j < sizeX)

}
