#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "SpatialFilter.h"

int main(int argc, char** argv) {
    std::cout << "--- Starting Noise Reduction Pipeline ---" << std::endl;

    std::string imagePath = "D:/tesy/test_image.jpg";
    cv::Mat cleanImage = cv::imread(imagePath, cv::IMREAD_COLOR);

    if (cleanImage.empty()) {
        std::cerr << "Error: Could not load image at " << imagePath << std::endl;
        return -1;
    }

    SpatialFilter filterEngine;

    std::cout << "Injecting synthetic Gaussian noise..." << std::endl;
    cv::Mat noisyImage = filterEngine.addGaussianNoise(cleanImage, 0.0, 30.0);

    // 1. Gaussian Filter
    std::cout << "Applying Gaussian Filter..." << std::endl;
    auto startGaussian = std::chrono::high_resolution_clock::now();
    cv::Mat gaussianResult = filterEngine.applyGaussianFilter(noisyImage, 7, 2.5);
    auto endGaussian = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gaussianTime = endGaussian - startGaussian;

    // 2. Bilateral Filter
    std::cout << "Applying Bilateral Filter..." << std::endl;
    auto startBilateral = std::chrono::high_resolution_clock::now();
    cv::Mat bilateralResult = filterEngine.applyBilateralFilter(noisyImage, 9, 75.0, 75.0);
    auto endBilateral = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> bilateralTime = endBilateral - startBilateral;

    // 3. AI Denoiser (DnCNN)
    std::cout << "Applying AI Denoiser (DnCNN)..." << std::endl;
    auto startAI = std::chrono::high_resolution_clock::now();
    // Make sure this path points to where you saved the exported ONNX file!
    cv::Mat aiResult = filterEngine.applyAIDenoiser(noisyImage, "C:/Users/khand/OneDrive/Documents/coding/noise reduction/models/dncnn_sidd_final.onnx");
    auto endAI = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> aiTime = endAI - startAI;

    // Print Telemetry
    std::cout << "\nPerformance Profile:" << std::endl;
    std::cout << "Gaussian processing time:  " << gaussianTime.count() << " ms" << std::endl;
    std::cout << "Bilateral processing time: " << bilateralTime.count() << " ms" << std::endl;
    std::cout << "AI DnCNN processing time:  " << aiTime.count() << " ms" << std::endl;

    // Save Outputs
    cv::imwrite("output_1_noisy.jpg", noisyImage);
    cv::imwrite("output_2_gaussian_blurred.jpg", gaussianResult);
    cv::imwrite("output_3_bilateral_filtered.jpg", bilateralResult);
    cv::imwrite("output_4_ai_denoised.jpg", aiResult);

    std::cout << "\nPipeline complete. Results saved to disk." << std::endl;

    return 0;
}