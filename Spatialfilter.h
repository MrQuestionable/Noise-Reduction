#pragma once

#include <opencv2/opencv.hpp>
#include <string>

class SpatialFilter {
public:
    SpatialFilter();
    ~SpatialFilter();

    /**
     * Adds synthetic Gaussian noise to an image to simulate sensor read noise.
     * @param src The clean input image.
     * @param mean The mean of the Gaussian distribution (usually 0).
     * @param stddev The standard deviation (intensity of the noise).
     * @return A new cv::Mat containing the noisy image.
     */
    cv::Mat addGaussianNoise(const cv::Mat& src, double mean = 0.0, double stddev = 25.0);

    /**
     * Applies a standard Gaussian Blur. 
     * Smooths noise but destroys edge details.
     * @param src The noisy input image.
     * @param kernelSize The size of the sliding matrix (e.g., 5 for a 5x5 matrix).
     * @param sigma The spatial variance.
     * @return The filtered image.
     */
    cv::Mat applyGaussianFilter(const cv::Mat& src, int kernelSize, double sigma);

    /**
     * 
     * Reduces noise while preserving sharp edges by factoring in pixel intensity differences.
     * @param src The noisy input image.
     * @param diameter Pixel neighborhood diameter.
     * @param sigmaColor Filter sigma in the color space (larger = mixes more distinct colors).
     * @param sigmaSpace Filter sigma in the coordinate space (larger = mixes pixels further away).
     * @return The filtered image.
     */
    cv::Mat applyBilateralFilter(const cv::Mat& src, int diameter, double sigmaColor, double sigmaSpace);

    /**
    * Peak Signal-to-Noise Ratio in dB (higher = closer to the original).
    */
    double calculatePSNR(const cv::Mat& original, const cv::Mat& processed);

    /**
     * Mean Structural Similarity Index, in [0, 1] (1 = identical).
     */
    double calculateSSIM(const cv::Mat& original, const cv::Mat& processed);
};