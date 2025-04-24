// Ryan Raad 2025
// CUDA header for normalizeAndTruncate

#ifndef NORMALIZE_AND_TRUNCATE_CUH
#define NORMALIZE_AND_TRUNCATE_CUH

#include "fhog.hpp"  // for CvLSVMFeatureMapCaskade, NUM_SECTOR, etc.

// Host‐callable entry point
int normalizeAndTruncateNaive(CvLSVMFeatureMapCaskade* map, const float alfa);

#endif // NORMALIZE_AND_TRUNCATE_CUH
