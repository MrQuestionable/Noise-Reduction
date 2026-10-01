#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "SpatialFilter.h"

int main(int argc, char** argv) {
    std::cout << "--- Starting Noise Reduction Pipeline ---" << std::endl;

    std::string imagePath = "D:/tesy/img9-1180x738.jpg"; 
    cv::Mat cleanImage = cv::imread(imagePath, cv::IMREAD_COLOR);

    if (cleanImage.empty()) {
        std::cerr << "Error: Could not load image at " << imagePath << std::endl;
        std::cerr << "Please place a 'test_image.jpg' in the directory you are running this from." << std::endl;
        return -1;
    }

    SpatialFilter filterEngine;

    std::cout << "Injecting synthetic Gaussian noise..." << std::endl;
    cv::Mat noisyImage = filterEngine.addGaussianNoise(cleanImage, 0.0, 30.0);

    std::cout << "Applying Gaussian Filter..." << std::endl;
    auto startGaussian = std::chrono::high_resolution_clock::now();
    cv::Mat gaussianResult = filterEngine.applyGaussianFilter(noisyImage, 7, 2.5);
    auto endGaussian = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gaussianTime = endGaussian - startGaussian;
    
    std::cout << "Applying Bilateral Filter..." << std::endl;
    auto startBilateral = std::chrono::high_resolution_clock::now();
    cv::Mat bilateralResult = filterEngine.applyBilateralFilter(noisyImage, 9, 75.0, 75.0);
    auto endBilateral = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> bilateralTime = endBilateral - startBilateral;

    std::cout << "\nPerformance Profile:" << std::endl;
    std::cout << "Gaussian processing time: " << gaussianTime.count() << " ms" << std::endl;
    std::cout << "Bilateral processing time: " << bilateralTime.count() << " ms" << std::endl;

    cv::imwrite("output_1_noisy.jpg", noisyImage);
    cv::imwrite("output_2_gaussian_blurred.jpg", gaussianResult);
    cv::imwrite("output_3_bilateral_filtered.jpg", bilateralResult);

    std::cout << "\nPipeline complete. Results saved to disk." << std::endl;

    return 0;
}