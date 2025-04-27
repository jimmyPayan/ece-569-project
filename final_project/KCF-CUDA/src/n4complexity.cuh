#ifndef N4_COMPLEXITY_CUH
#define N4_COMPLEXITY_CUH

#include "fhog.hpp"

void featureGPU(int sizeY, int sizeX, int k, int height, int width, int numFeatures, float *map, int stringSize, int *alfa, float *r, float *w, int *nearest);
void getFeatureMapsGPU(const IplImage* image, const int k, CvLSVMFeatureMapCaskade **map);




#endif
