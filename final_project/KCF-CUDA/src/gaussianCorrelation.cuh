// Team1 : Alex Caulin-Cardo
// Gaussian Correlation header to be called from runtracker.cpp
#ifndef GAUSSIAN_CORRELATION_CUH
#define GAUSSIAN_CORRELATION_CUH

#include <opencv2/core.hpp>
#include <cufft.h>

// Workspace Struct Definition
struct GaussianCorrelationWorkspace {
    float* d_x1;
    float* d_x2;
    cufftComplex* d_x1f;
    cufftComplex* d_x2f;
    cufftComplex* d_mult;
    float* d_ifft;
    float* d_result;
    cufftHandle plan_fwd;
    cufftHandle plan_inv;
    int size_x;
    int size_y;
    int size_z;
};

// Funcs for 1 mem allocation and free
void initCUDAMem(GaussianCorrelationWorkspace& ws, int size_y, int size_x, int size_z);
void freeCUDAMem(GaussianCorrelationWorkspace& ws);

// CUDA Gaussian Correlation entry point
cv::Mat gaussianCorrelationGPU(const cv::Mat& x1, const cv::Mat& x2,
                                int size_y, int size_x, int size_z, float sigma,
                                GaussianCorrelationWorkspace& ws);

#endif
