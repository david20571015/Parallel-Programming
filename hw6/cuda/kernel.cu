#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "hostFE.h"

__global__ void convolutionKernel(int filterWidth, float* filter,
                                  int imageHeight, int imageWidth,
                                  float* inputImage, float* outputImage) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  int halffilterSize = filterWidth / 2;
  float sum = 0.0f;
  int filterIdx = -1;
  int x, y;

  for (int k = -halffilterSize; k <= halffilterSize; k++) {
    x = i + k;
    if (x < 0 || x >= imageHeight) {
      filterIdx += filterWidth;
      continue;
    }
    for (int l = -halffilterSize; l <= halffilterSize; l++) {
      y = j + l;
      if (y < 0 || y >= imageWidth) {
        ++filterIdx;
        continue;
      }

      if (filter[++filterIdx] == 0) continue;
      int imageIdx = x * imageWidth + y;
      sum += filter[filterIdx] * inputImage[imageIdx];
    }
  }

  outputImage[i * imageWidth + j] = sum;
}

void hostFE(int filterWidth, float* filter, int imageHeight, int imageWidth,
            float* inputImage, float* outputImage) {
  int filterSize = filterWidth * filterWidth;
  int imageSize = imageHeight * imageWidth;

  // Allocate memory on the device and copy the data to the device
  float* dFilter;
  cudaMalloc((void**)&dFilter, sizeof(float) * filterSize);
  float* dInputImage;
  cudaMalloc((void**)&dInputImage, sizeof(float) * imageSize);
  float* dOutoutImage;
  cudaMalloc((void**)&dOutoutImage, sizeof(float) * imageSize);

  cudaMemcpy(dInputImage, inputImage, sizeof(float) * imageSize,
             cudaMemcpyHostToDevice);
  cudaMemcpy(dFilter, filter, sizeof(float) * filterSize,
             cudaMemcpyHostToDevice);

  constexpr int BLOCK_SIZE = 8;
  dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_size(imageHeight / BLOCK_SIZE, imageWidth / BLOCK_SIZE);

  convolutionKernel<<<grid_size, block_size>>>(
      filterWidth, dFilter, imageHeight, imageWidth, dInputImage, dOutoutImage);

  cudaMemcpy(outputImage, dOutoutImage, sizeof(float) * imageSize,
             cudaMemcpyDeviceToHost);

  cudaFree(dFilter);
  cudaFree(dInputImage);
  cudaFree(dOutoutImage);
}