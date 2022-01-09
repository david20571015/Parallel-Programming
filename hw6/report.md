# Parallel Programming HW6

###### tags: `Parallel Programming`

## Q1

> Explain your implementation. How do you optimize the performance of convolution?

* hostEF.h
  1. Created buffer by `CL_MEM_USE_HOST_PTR` flag which maintained a reference to that memory area and depending on the implementation it might access it directly while kernels were executing or it might cache it. Thus,  we didn't need to copy the data to the device by `clEnqueueWriteBuffer`.

  ```c=
  // Create memory buffers on the device for each array
  cl_mem filterBuffer =
      clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                     sizeof(float) * filterSize, filter, &status);
  cl_mem inputImageBuffer =
      clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                     sizeof(float) * imageSize, inputImage, &status);
  cl_mem outputImageBuffer =
      clCreateBuffer(*context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                     sizeof(float) * imageSize, outputImage, &status);

  // Copy the data to the device
  // status = clEnqueueWriteBuffer(commandQueue, filterBuffer, CL_TRUE, 0,
  //                               sizeof(float) * filterSize, (void *)filter,
  //                               0, NULL, NULL);
  // status = clEnqueueWriteBuffer(commandQueue, inputImageBuffer, CL_TRUE, 0,
  //                               sizeof(float) * imageSize, (void
  //                               *)inputImage, 0, NULL, NULL);
  ```

* kernel.cl
  1. Used `__constant` type qualifier to store filter in cache of device.
  
  ```c=
  __kernel void convolution(int filterWidth, __constant float *filter,
                            int imageHeight, int imageWidth,
                            __global const float *inputImage,
                            __global float *outputImage);
  ```

  2. Reduced computation by relocated the condition statement.  
  3. Stored the position of the pixel in the image to avoid duplicate computation.
  4. Computed the position of the pixel in the filter by add its index.

  ```c=
  // origin
  for (k = -halffilterSize; k <= halffilterSize; k++) {
    for (l = -halffilterSize; l <= halffilterSize; l++) {
      if (i + k >= 0 && i + k < imageHeight && j + l >= 0 &&
          j + l < imageWidth) {
        sum +=
            inputImage[(i + k) * imageWidth + j + l] *
            filter[(k + halffilterSize) * filterWidth + l + halffilterSize];
      }
    }
  }

  // modified
  int filterIdx = -1;
  int2 pos;

  for (int k = -halffilterSize; k <= halffilterSize; k++) {
    pos.x = i + k;
    if (pos.x < 0 || pos.x >= imageHeight) {
      filterIdx += filterWidth;
      continue;
    }
    for (int l = -halffilterSize; l <= halffilterSize; l++) {
      pos.y = j + l;
      if (pos.y < 0 || pos.y >= imageWidth) {
        ++filterIdx;
        continue;
      }

      if (filter[++filterIdx] == 0) continue;
      int imageIdx = pos.x * imageWidth + pos.y;
      sum += filter[filterIdx] * inputImage[imageIdx];
    }
  }
  ```

* Reference
  * [CL_MEM_USE_HOST_PTR](https://stackoverflow.com/questions/25496656/cl-mem-use-host-ptr-vs-cl-mem-copy-host-ptr-vs-cl-mem-alloc-host-ptr)
  * [clCreateBuffer](https://man.opencl.org/clCreateBuffer.html?fbclid=IwAR2YjajINvaNXaKykiV7fxYmsMYfFbvi8rxP5RwIJ0z9xXSE6eIfkCwFc7Q#:~:text=OpenCL%20implementations%20are%20allowed%20to%20cache%20the%20buffer%20contents%20pointed%20to%20by%20host_ptr%20in%20device%20memory.%20This%20cached%20copy%20can%20be%20used%20when%20kernels%20are%20executed%20on%20a%20device.)

## Q2

> Rewrite the program using CUDA. (1) Explain your CUDA implementation, (2) plot a chart to show the performance difference between using OpenCL and CUDA, and (3) explain the result.

1. Used the same algorithm as OpenCL version.

2. The CUDA version was faster than the OpenCL version in all filter sizes.
  ![opencl vs cuda](https://i.imgur.com/d9STTMu.png)

3. OpenCL traded off speed for compatibility between different platforms. Otherwise, CUDA was designed for NVIDIA's GPU; therefore, it might perform better than OpenCL.
