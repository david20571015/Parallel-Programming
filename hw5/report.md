# Parallel Programming HW5

###### tags: `Parallel Programming`

## Q1

> What are the pros and cons of the three methods? Give an assumption about their performances.

* Method 1
  * Pros
    * Each thread only needs to compute one pixel. The GPU has a large number of threads for computation.
    * It uses less memory bacause `malloc` creates pageable memory.
  * Cons
    * The pageable host memory might be written into swap by the OS. Therefore, this method might take more time to transfer data.

* Method 2
  * Pros
    * Each thread only needs to compute one pixel. The GPU has a large number of threads for computation.
    * Cuda driver can directly transfer data between device memory and pinned host memory created by `cudaHostAlloc`.
    * The device memory is aligned by `cudaMallocPitch` which conduces to faster array access.
  * Cons
    * More memory cost due to the pinned memory, and it might reduce the host performance.
    * To align the device memory, there might be some memory wasted for padding.

* Method 3
  * Pros
    * Because each thread compute a group of pixels, we can save the computational resource and reduce the overhead of launching a new thread.
    * Cuda driver can directly transfer data between device memory and pinned host memory created by `cudaHostAlloc`.
    * The device memory is aligned by `cudaMallocPitch` which conduces to faster array access.
  * Cons
    * We need to determine the grouping strategy and the group size.
    * The speed is easily slow down by the unbalance work load between each thread.
    * This method might lead to low GPU utility.
    * The workloads between each thread of a block might be more unbalance.

* Assumption
  * Method 2 > Method 1
    The memory used by this task is 1200 * 1400 * 4 bytes ~= 7.3 MBs, which is about 0.1% of the total memory (6GB) on the GPU. Hence, we don't need to worry about the memory usage.
  * Method 1 > Method 3
    Low GPU utility will reduce the performance.

* Reference

  * [How to Optimize Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)

## Q2

> How are the performances of the three methods? Plot a chart to show the differences among the three methods.

> The group size in method 3 is 64000.

* View 1
![View 1](https://i.imgur.com/uFCfaSP.png)

<!-- | view 1 | 1       | 2       | 3       |
| :----: | :-----: | :-----: | :-----: |
| 1000   | 7.006   | 8.569   | 9.530   |
| 10000  | 33.437  | 33.826  | 38.536  |
| 100000 | 305.842 | 307.383 | 339.542 | -->

* View 2
![View 2](https://i.imgur.com/XhG5lbc.png)

<!-- | view 1 | 1      | 2      | 3      |
| :----: | :----: | :----: | :----: |
| 1000   | 4.152  | 6.106  | 7.144  |
| 10000  | 7.038  | 9.144  | 10.249 |
| 100000 | 29.744 | 30.353 | 38.305 | -->

## Q3

> Explain the performance differences thoroughly based on your experimental results. Does the results match your assumption? Why or why not.

The result was **Method 1 > Method 2 > Method 3** in each case, it didn't match my assumption.

To reveal the reason, I commented the `mandelbrotThreadRef` in main.cpp and used `nvprof ./mandelbrot -g 1 -v 1 -i 10000` to compare the time cost of API calls.

In method 1 and 2, the difference between both methods were

* the way to allocate memory on host
  * Method 1 (malloc) : 0.026ms
  * Method 2 (cudaHostAlloc) : **156.48ms**
* the way to allocate memory on device
  * Method 1 (cudaMalloc) : **115.01ms**
  * Method 2 (cudaMallocPitch) : 1.0347ms
* the way to copy data from device to host
  * Method 1 (cudaMemcpy) : **328.08ms**
  * Method 2 (cudaMemcpy2D) : 297.50ms
* data alignment on device (speed of mandelKernel)
  * Method 1 (No) : **308.72ms**
  * Method 2 (Yes) : 293.47ms

Obviously, method 2 spent much more time on allocated memory on host than method 1 allocated memory on device.

> > nvprof cannot collect the information of host API calls (malloc). So I use CycleTimer to profile the time cost of `malloc`. [stackoverflow](https://stackoverflow.com/questions/56658676/how-to-get-malloc-to-show-up-in-nvprofs-statistical-profiler)

<!-- ![1](https://i.imgur.com/cr4GnND.png)
![2](https://i.imgur.com/w6B4TiS.png) -->

## Q4

> Can we do even better? Think a better approach and explain it. Implement your method in `kernel4.cu`.

* I used method 2, but directly copy memory from device to `*img` to reduce unnecessary memory copy.

```c++
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int* img,
            int resX, int resY, int maxIterations) {
  float stepX = (upperX - lowerX) / resX;
  float stepY = (upperY - lowerY) / resY;

  constexpr int BLOCK_SIZE = 8;
  dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_size(resX / BLOCK_SIZE, resY / BLOCK_SIZE);

  // Allocate memory on the device
  int* d_img;
  size_t pitch;
  cudaMallocPitch((void**)&d_img, &pitch, sizeof(int) * resX, resY);

  // Launch the kernel
  mandelKernel<<<grid_size, block_size>>>(lowerX, lowerY, d_img, stepX, stepY,
                                          maxIterations, pitch, resY);

  // Copy the result from the device to the host
  cudaMemcpy2D(img, sizeof(int) * resX, d_img, pitch, sizeof(int) * resX, resY,
               cudaMemcpyDeviceToHost);
  cudaFree(d_img);
}
```
