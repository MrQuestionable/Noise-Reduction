# Noise Reduction

![Build](https://github.com/<your-username>/Noise-Reduction/actions/workflows/<workflow-file>.yml/badge.svg)
![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.12.0-green)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)

A C++ image noise reduction project built on OpenCV. It simulates sensor noise, applies spatial filters to remove it, and measures the quality of the result with objective metrics (PSNR and SSIM). The metrics are verified by an automated Google Test suite that runs on every push through GitHub Actions.

## Features

- **Synthetic noise generation:** adds zero-mean Gaussian noise to simulate sensor read noise. The noise is generated in floating point and saturated back to 8-bit, so it is not clipped asymmetrically.
- **Gaussian filter:** a standard Gaussian blur. It smooths noise effectively but also softens edges.
- **Bilateral filter:** an edge-preserving filter that weighs both spatial distance and intensity difference.
- **Quality metrics:**
  - **PSNR** (Peak Signal-to-Noise Ratio, in dB): higher means closer to the original.
  - **SSIM** (Structural Similarity Index, 0 to 1): measures preservation of structure, where 1 is identical.
- **Automated testing:** Google Test cases that check the filters and metrics, with a fixed random seed so results are reproducible.
- **CI/CD:** a GitHub Actions pipeline that builds the project, runs the tests, and uploads the executable as an artifact.

## Project Structure

```
Noise-Reduction/
├── SpatialFilter.h        # SpatialFilter class declaration
├── SpatialFilter.cpp      # Noise, filters, PSNR and SSIM implementations
├── main.cpp               # Application entry point
├── tests/
│   └── test_filters.cpp   # Google Test suite
├── CMakeLists.txt         # Build configuration
└── .github/workflows/     # GitHub Actions CI pipeline
```

## API Overview

All functionality lives in the `SpatialFilter` class.

```cpp
SpatialFilter filter;

// Add Gaussian noise (mean, standard deviation)
cv::Mat noisy = filter.addGaussianNoise(clean, 0.0, 30.0);

// Denoise
cv::Mat gaussian  = filter.applyGaussianFilter(noisy, 5, 1.5);
cv::Mat bilateral = filter.applyBilateralFilter(noisy, 9, 200.0, 5.0);

// Evaluate against the clean reference
double psnr = filter.calculatePSNR(clean, bilateral);
double ssim = filter.calculateSSIM(clean, bilateral);
```

| Method | Description |
|---|---|
| `addGaussianNoise(src, mean, stddev)` | Returns a noisy copy of `src` |
| `applyGaussianFilter(src, kernelSize, sigma)` | Gaussian blur; even kernel sizes are bumped to the next odd value |
| `applyBilateralFilter(src, diameter, sigmaColor, sigmaSpace)` | Edge-preserving smoothing |
| `calculatePSNR(original, processed)` | PSNR in dB |
| `calculateSSIM(original, processed)` | Mean SSIM, clamped to [0, 1] |

> **Tip:** OpenCV's bilateral filter sums color differences across all channels on color images, so `sigmaColor` usually needs to be set higher for color images than for grayscale ones.

## Requirements

- Windows with Visual Studio 2022 or newer (MSVC)
- [CMake](https://cmake.org/) 3.14 or newer
- [OpenCV](https://opencv.org/) 4.12.0 (prebuilt Windows package)
- Internet access on first configure, because Google Test is fetched automatically with `FetchContent`

## Building

1. Install OpenCV and note its location (for example `C:/opt/opencv`).
2. Add the OpenCV `bin` folder (for example `C:\opt\opencv\build\x64\vc16\bin`) to your `PATH` so the executables can find the OpenCV DLL at runtime.
3. Configure and build:

```bash
cmake -B build -G Ninja -DOpenCV_DIR="C:/opt/opencv/build/x64/vc16/lib"
cmake --build build
```

If `OpenCV_DIR` is not passed, `CMakeLists.txt` falls back to a default path for local development. Edit that default to match your install.

You can also open the folder directly in Visual Studio, which handles configuration through CMake.

## Running the Tests

```bash
cd build
ctest --output-on-failure
```

Or run the test executable directly for more detailed output:

```bash
./FilterTests.exe
```

The suite covers three behaviors:

| Test | What it verifies |
|---|---|
| `GaussianFilterImprovesPSNR` | Gaussian filtering raises PSNR relative to the noisy image |
| `BilateralFilterPreservesStructureBetterThanGaussian` | The bilateral filter achieves a higher SSIM than a Gaussian filter on an image with a sharp edge |
| `SSIMIsWithinValidRange` | SSIM always falls within [0, 1] |

The noise generator uses a fixed seed, so every run produces identical results and the tests are not flaky.

## Continuous Integration

The workflow in `.github/workflows/` runs on every push and pull request to `main` or `master`. It:

1. Checks out the code
2. Restores OpenCV from cache (or downloads and extracts it)
3. Configures and builds the project with CMake and MSVC in Release mode
4. Runs the Google Test suite with CTest
5. Uploads `noise_reduction.exe` as a build artifact

## How the Metrics Work

- **PSNR** is computed from the mean squared error between two images: `PSNR = 10 · log10(255² / MSE)`. It is a simple pixel-wise measure.
- **SSIM** compares local luminance, contrast and structure using an 11×11 Gaussian window (σ = 1.5), averaged across color channels. It correlates better with perceived quality than PSNR, especially around edges.

## Roadmap

- Median and non-local means filters
- Support for loading and saving user-provided images from the command line
- Cross-platform CI (Linux)

