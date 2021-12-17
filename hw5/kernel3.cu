#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__device__ int mandel(float c_re, float c_im, int count) {
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < count; ++i) {
    if (z_re * z_re + z_im * z_im > 4.f) break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }

  return i;
}

__global__ void mandelKernel(float lowerX, float lowerY, int* img, float stepX,
                             float stepY, int resX, int resY, int maxIterations,
                             int group_size) {
  int thisX = blockIdx.x * blockDim.x + threadIdx.x;
  int thisY = blockIdx.y * blockDim.y + threadIdx.y;

  int group = thisY * resX + thisX;
  if (group >= group_size) return;

  for (int i = group; i < resX * resY; i += group_size) {
    thisX = i % resX;
    thisY = i / resX;
    float x = lowerX + stepX * thisX;
    float y = lowerY + stepY * thisY;

    img[i] = mandel(x, y, maxIterations);
  }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int* img,
            int resX, int resY, int maxIterations) {
  float stepX = (upperX - lowerX) / resX;
  float stepY = (upperY - lowerY) / resY;

  static constexpr int BLOCK_SIZE = 16;
  dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_size(resX / BLOCK_SIZE, resY / BLOCK_SIZE);

  // Allocate memory on the host
  int* h_img;
  cudaHostAlloc((void**)&h_img, sizeof(int) * resX * resY,
                cudaHostAllocDefault);

  // Allocate memory on the device
  int* d_img;
  size_t pitch;
  cudaMallocPitch((void**)&d_img, &pitch, resX * resY, sizeof(int));

  // Launch the kernel
  static int GROUP_SIZE = 64000;
  mandelKernel<<<grid_size, block_size>>>(lowerX, lowerY, d_img, stepX, stepY,
                                          resX, resY, maxIterations,
                                          GROUP_SIZE);

  // Copy the result from the device to the host
  cudaMemcpy(h_img, d_img, sizeof(int) * resX * resY, cudaMemcpyDeviceToHost);
  cudaFree(d_img);

  memcpy(img, h_img, sizeof(int) * resX * resY);
  cudaFreeHost(h_img);
}
