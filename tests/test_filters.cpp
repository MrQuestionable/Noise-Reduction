#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "SpatialFilter.h"

class FilterTest : public ::testing::Test {
protected:
    cv::Mat cleanImage;
    cv::Mat noisyImage;
    SpatialFilter filter;

    void SetUp() override {
        cleanImage = cv::Mat(100, 100, CV_8UC3, cv::Scalar(64, 64, 64));
        cv::rectangle(cleanImage, cv::Point(25, 25), cv::Point(75, 75),
            cv::Scalar(192, 192, 192), cv::FILLED);

        noisyImage = filter.addGaussianNoise(cleanImage, 0.0, 30.0);
    }
};

TEST_F(FilterTest, GaussianFilterImprovesPSNR) {
    cv::Mat filtered = filter.applyGaussianFilter(noisyImage, 5, 1.5);

    double noisyPSNR = filter.calculatePSNR(cleanImage, noisyImage);
    double filteredPSNR = filter.calculatePSNR(cleanImage, filtered);

    EXPECT_GT(filteredPSNR, noisyPSNR);
}

TEST_F(FilterTest, BilateralFilterPreservesStructureBetterThanGaussian) {
    cv::Mat gaussian = filter.applyGaussianFilter(noisyImage, 9, 2.5);
    cv::Mat bilateral = filter.applyBilateralFilter(noisyImage, 9, 200.0, 5.0);

    double gSSIM = filter.calculateSSIM(cleanImage, gaussian);
    double bSSIM = filter.calculateSSIM(cleanImage, bilateral);
    std::cout << "Bilateral SSIM: " << bSSIM << " vs Gaussian SSIM: " << gSSIM << std::endl;

    EXPECT_GT(bSSIM, gSSIM);
}

TEST_F(FilterTest, SSIMIsWithinValidRange) {
    cv::Mat filtered = filter.applyGaussianFilter(noisyImage, 5, 1.5);
    double ssim = filter.calculateSSIM(cleanImage, filtered);

    EXPECT_GE(ssim, 0.0);
    EXPECT_LE(ssim, 1.0);

}
TEST_F(FilterTest, AIDenoiserOutperformsBilateralOnRealNoise) {
    // For this specific test, load a real noisy image from the SIDD dataset
    cv::Mat realNoisyImage = cv::imread("tests/data/sidd_noisy_sample.png", cv::IMREAD_COLOR);
    cv::Mat realCleanImage = cv::imread("tests/data/sidd_ground_truth.png", cv::IMREAD_COLOR);

    ASSERT_FALSE(realNoisyImage.empty()) << "Failed to load SIDD noisy sample";

    // Apply filters
    cv::Mat bilateral = filter.applyBilateralFilter(realNoisyImage, 9, 75.0, 75.0);
    cv::Mat aiDenoised = filter.applyAIDenoiser(realNoisyImage, "models/dncnn_denoiser.onnx");

    // Calculate metrics
    double bPSNR = filter.calculatePSNR(realCleanImage, bilateral);
    double aiPSNR = filter.calculatePSNR(realCleanImage, aiDenoised);

    // The AI model should mathematically reconstruct real sensor noise better than Bilateral
    EXPECT_GT(aiPSNR, bPSNR);
}