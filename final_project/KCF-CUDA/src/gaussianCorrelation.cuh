// Team1 : Alex Caulin-Cardo
// Gaussian Correlation header to be called from runtracker.cpp
#ifndef GAUSSIAN_CORRELATION_CUH
#define GAUSSIAN_CORRELATION_CUH

#include <opencv2/core.hpp>

// CUDA Gaussian Correlation entry point
cv::Mat gaussianCorrelationGPU(const cv::Mat& x1, const cv::Mat& x2, int size_y, int size_x, int size_z, float sigma);

#endif

