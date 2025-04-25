#pragma once
#include "fhog.hpp"   // for CvLSVMFeatureMapCaskade, LATENT_SVM_OK

// host‐callable wrapper
int PCAFeatureMapsGPU(CvLSVMFeatureMapCaskade *map);
