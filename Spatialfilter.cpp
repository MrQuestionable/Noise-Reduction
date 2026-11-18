#include "SpatialFilter.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <opencv2/dnn.hpp>

SpatialFilter::SpatialFilter() {}
SpatialFilter::~SpatialFilter() {}

cv::Mat SpatialFilter::addGaussianNoise(const cv::Mat& src, double mean, double stddev) {
    // Fixed seed so tests are reproducible
    cv::RNG rng(12345);

    // Signed float noise (an 8-bit Mat would clamp all negative values to 0)
    cv::Mat srcFloat, noise(src.size(), CV_32FC(src.channels()));
    src.convertTo(srcFloat, CV_32F);
    rng.fill(noise, cv::RNG::NORMAL, mean, stddev);

    cv::Mat noisyFloat = srcFloat + noise;

    cv::Mat noisyImage;
    noisyFloat.convertTo(noisyImage, src.type()); // saturates to 0..255
    return noisyImage;
}

cv::Mat SpatialFilter::applyGaussianFilter(const cv::Mat& src, int kernelSize, double sigma) {
    cv::Mat result;

    if (kernelSize % 2 == 0) kernelSize++;

    cv::GaussianBlur(src, result, cv::Size(kernelSize, kernelSize), sigma, sigma);
    return result;
}

cv::Mat SpatialFilter::applyBilateralFilter(const cv::Mat& src, int diameter, double sigmaColor, double sigmaSpace) {
    cv::Mat result;
    cv::bilateralFilter(src, result, diameter, sigmaColor, sigmaSpace);
    return result;
}

double SpatialFilter::calculatePSNR(const cv::Mat& original, const cv::Mat& processed) {
    CV_Assert(original.size() == processed.size() && original.type() == processed.type());

    cv::Mat diff;
    cv::absdiff(original, processed, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);

    cv::Scalar s = cv::sum(diff);
    double sse = s[0] + s[1] + s[2] + s[3];
    if (sse <= 1e-10) return 100.0; // identical images

    double mse = sse / static_cast<double>(original.channels() * original.total());
    return 10.0 * std::log10((255.0 * 255.0) / mse);
}

double SpatialFilter::calculateSSIM(const cv::Mat& original, const cv::Mat& processed) {
    CV_Assert(original.size() == processed.size() && original.type() == processed.type());

    const double C1 = 6.5025;   // (0.01 * 255)^2
    const double C2 = 58.5225;  // (0.03 * 255)^2

    cv::Mat I1, I2;
    original.convertTo(I1, CV_32F);
    processed.convertTo(I2, CV_32F);

    cv::Mat I1_2 = I1.mul(I1);
    cv::Mat I2_2 = I2.mul(I2);
    cv::Mat I1_I2 = I1.mul(I2);

    cv::Mat mu1, mu2, sigma1_2, sigma2_2, sigma12;
    cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    cv::Mat t1 = 2 * mu1_mu2 + cv::Scalar::all(C1);
    cv::Mat t2 = 2 * sigma12 + cv::Scalar::all(C2);
    cv::Mat numerator = t1.mul(t2);

    t1 = mu1_2 + mu2_2 + cv::Scalar::all(C1);
    t2 = sigma1_2 + sigma2_2 + cv::Scalar::all(C2);
    cv::Mat denominator = t1.mul(t2);

    cv::Mat ssim_map;
    cv::divide(numerator, denominator, ssim_map);

    cv::Scalar m = cv::mean(ssim_map);
    double ssim = 0.0;
    for (int c = 0; c < original.channels(); ++c) ssim += m[c];
    ssim /= original.channels();

    return std::max(0.0, std::min(1.0, ssim));
}
cv::Mat SpatialFilter::applyAIDenoiser(const cv::Mat& noisyImage, const std::string& modelPath) {
    // 1. Load the ONNX model from disk
    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);

    // 2. Convert standard OpenCV Mat (HWC, BGR, 0-255) to AI Tensor Blob (NCHW, RGB, 0.0-1.0)
    // The 'true' parameter automatically swaps BGR to RGB for the AI
    cv::Mat blob = cv::dnn::blobFromImage(noisyImage, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false);

    // 3. Run the image through the DnCNN
    net.setInput(blob);
    cv::Mat outputBlob = net.forward();

    // 4. Convert the AI Tensor output back to a standard OpenCV image
    std::vector<int> sz = { outputBlob.size[2], outputBlob.size[3] };
    cv::Mat channels[3];
    for (int i = 0; i < 3; ++i) {
        cv::Mat channel(static_cast<int>(sz.size()), sz.data(), CV_32F, (void*)outputBlob.ptr<float>(0, i));
        channels[i] = channel.clone();
    }

    cv::Mat cleanFloat, finalCleanRGB, finalCleanBGR;
    cv::merge(channels, 3, cleanFloat);

    // Convert 0.0-1.0 float back to standard 0-255 pixels
    cleanFloat.convertTo(finalCleanRGB, CV_8UC3, 255.0);

    // 5. The AI outputs RGB, so we must swap it back to OpenCV's default BGR
    cv::cvtColor(finalCleanRGB, finalCleanBGR, cv::COLOR_RGB2BGR);

    return finalCleanBGR;
}