#include "hostFE.h"

#include <stdio.h>
#include <stdlib.h>

#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program) {
  int filterSize = filterWidth * filterWidth;
  int imageSize = imageHeight * imageWidth;

  cl_int status;

  // Create command queue
  cl_command_queue commandQueue =
      clCreateCommandQueue(*context, *device, 0, &status);
  // CHECK(status, "clCreateCommandQueue");

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
  // CHECK(status, "clCreateBuffer");

  // Copy the data to the device
  // status = clEnqueueWriteBuffer(commandQueue, filterBuffer, CL_TRUE, 0,
  //                               sizeof(float) * filterSize, (void *)filter,
  //                               0, NULL, NULL);
  // status = clEnqueueWriteBuffer(commandQueue, inputImageBuffer, CL_TRUE, 0,
  //                               sizeof(float) * imageSize, (void
  //                               *)inputImage, 0, NULL, NULL);
  // CHECK(status, "clEnqueueWriteBuffer");

  // Set kernel function and arguments
  cl_kernel kernel = clCreateKernel(*program, "convolution", &status);
  clSetKernelArg(kernel, 0, sizeof(int), (void *)&filterWidth);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&filterBuffer);
  clSetKernelArg(kernel, 2, sizeof(int), (void *)&imageHeight);
  clSetKernelArg(kernel, 3, sizeof(int), (void *)&imageWidth);
  clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&inputImageBuffer);
  clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&outputImageBuffer);
  // CHECK(status, "clSetKernelArg");

  // Set the work-item dimensions
  size_t localWorkSize[2] = {10, 10};
  size_t globalWorkSize[2] = {imageHeight, imageWidth};

  // Execute the kernel
  status = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSize,
                                  localWorkSize, 0, NULL, NULL);
  // CHECK(status, "clEnqueueNDRangeKernel");

  // Read the output data
  status = clEnqueueReadBuffer(commandQueue, outputImageBuffer, CL_TRUE, 0,
                               sizeof(float) * imageSize, (void *)outputImage,
                               0, NULL, NULL);
  // CHECK(status, "clEnqueueReadBuffer");D
}

// void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
//             float *inputImage, float *outputImage, cl_device_id *device,
//             cl_context *context, cl_program *program) {
//   int filterSize = filterWidth * filterWidth;
//   int imageSize = imageHeight * imageWidth;

//   cl_int status;

//   // Create command queue
//   cl_command_queue commandQueue =
//       clCreateCommandQueue(*context, *device, 0, &status);

//   // Create pinned memory buffers on the host
//   cl_mem filterPinnedBuffer =
//       clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
//                      sizeof(float) * filterSize, NULL, &status);
//   cl_mem inputImagePinnedBuffer =
//       clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
//                      sizeof(float) * imageSize, NULL, &status);
//   cl_mem outputImagePinnedBuffer =
//       clCreateBuffer(*context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
//                      sizeof(float) * imageSize, NULL, &status);

//   // Map standard pointers to reference the pinned memory buffers
//   float *filterData = (float *)clEnqueueMapBuffer(
//       commandQueue, filterPinnedBuffer, CL_TRUE, CL_MAP_WRITE, 0,
//       sizeof(float) * filterSize, 0, NULL, NULL, &status);
//   float *inputImageData = (float *)clEnqueueMapBuffer(
//       commandQueue, inputImagePinnedBuffer, CL_TRUE, CL_MAP_WRITE, 0,
//       sizeof(float) * imageSize, 0, NULL, NULL, &status);
//   float *outputImageData = (float *)clEnqueueMapBuffer(
//       commandQueue, outputImagePinnedBuffer, CL_TRUE, CL_MAP_READ, 0,
//       sizeof(float) * imageSize, 0, NULL, NULL, &status);

//   // Copy the data to the pinned memory buffers
//   memcpy(filterData, filter, sizeof(float) * filterSize);
//   memcpy(inputImageData, inputImage, sizeof(float) * imageSize);

//   // Create memory buffers on the device
//   cl_mem filterBuffer = clCreateBuffer(
//       *context, CL_MEM_READ_ONLY, sizeof(float) * filterSize, NULL, &status);
//   cl_mem inputImageBuffer = clCreateBuffer(
//       *context, CL_MEM_READ_ONLY, sizeof(float) * imageSize, NULL, &status);
//   cl_mem outputImageBuffer = clCreateBuffer(
//       *context, CL_MEM_WRITE_ONLY, sizeof(float) * imageSize, NULL, &status);

//   // Copy the data to the device
//   status = clEnqueueWriteBuffer(commandQueue, filterBuffer, CL_TRUE, 0,
//                                 sizeof(float) * filterSize, (void
//                                 *)filterData, 0, NULL, NULL);
//   status = clEnqueueWriteBuffer(commandQueue, inputImageBuffer, CL_TRUE, 0,
//                                 sizeof(float) * imageSize,
//                                 (void *)inputImageData, 0, NULL, NULL);

//   // Set kernel function and arguments
//   cl_kernel kernel = clCreateKernel(*program, "convolution", &status);
//   clSetKernelArg(kernel, 0, sizeof(int), (void *)&filterWidth);
//   clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&filterBuffer);
//   clSetKernelArg(kernel, 2, sizeof(int), (void *)&imageHeight);
//   clSetKernelArg(kernel, 3, sizeof(int), (void *)&imageWidth);
//   clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&inputImageBuffer);
//   clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&outputImageBuffer);

//   // Set the work-item dimensions
//   size_t localWorkSize[2] = {10, 10};
//   size_t globalWorkSize[2] = {imageHeight, imageWidth};

//   // Execute the kernel
//   status = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL,
//   globalWorkSize,
//                                   localWorkSize, 0, NULL, NULL);

//   // Read the output data
//   status = clEnqueueReadBuffer(commandQueue, outputImageBuffer, CL_TRUE, 0,
//                                sizeof(float) * imageSize,
//                                (void *)outputImageData, 0, NULL, NULL);
//   memcpy(outputImage, outputImageData, sizeof(float) * imageSize);
// }

// void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
//             float *inputImage, float *outputImage, cl_device_id *device,
//             cl_context *context, cl_program *program) {
//   int filterSize = filterWidth * filterWidth;
//   int imageSize = imageHeight * imageWidth;

//   cl_int status;

//   // Create command queue
//   cl_command_queue commandQueue =
//       clCreateCommandQueue(*context, *device, 0, &status);
//   // CHECK(status, "clCreateCommandQueue");

//   // Create memory buffers on the device for each array
//   cl_mem filterBuffer =
//       clCreateBuffer(*context, CL_MEM_USE_HOST_PTR, sizeof(float) *
//       filterSize,
//                      filter, &status);

//   cl_image_format imageFormat = {.image_channel_order = CL_R,
//                                  .image_channel_data_type = CL_FLOAT};
//   cl_image_desc imageDesc = {.image_type = CL_MEM_OBJECT_IMAGE2D,
//                              .image_width = imageWidth,
//                              .image_height = imageHeight};

//   cl_mem inputImageBuffer =
//       clCreateImage(*context, CL_MEM_USE_HOST_PTR, &imageFormat, &imageDesc,
//                     inputImage, &status);
//   cl_mem outputImageBuffer =
//       clCreateImage(*context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
//                     &imageFormat, &imageDesc, outputImage, &status);

//   // CHECK(status, "clCreateBuffer");

//   // Copy the data to the device
//   status = clEnqueueWriteBuffer(commandQueue, filterBuffer, CL_TRUE, 0,
//                                 sizeof(float) * filterSize, (void *)filter,
//                                 0, NULL, NULL);
//   status = clEnqueueWriteImage(
//       commandQueue, inputImageBuffer, CL_TRUE, (size_t[3]){0, 0, 0},
//       (size_t[3]){imageWidth, imageHeight, 1}, 0, 0, inputImage, 0, NULL,
//       NULL);

//   // CHECK(status, "clEnqueueWriteBuffer");

//   // Set kernel function and arguments
//   cl_kernel kernel = clCreateKernel(*program, "convolution", &status);
//   clSetKernelArg(kernel, 0, sizeof(int), (void *)&filterWidth);
//   clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&filterBuffer);
//   clSetKernelArg(kernel, 2, sizeof(int), (void *)&imageHeight);
//   clSetKernelArg(kernel, 3, sizeof(int), (void *)&imageWidth);
//   clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&inputImageBuffer);
//   clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&outputImageBuffer);
//   // CHECK(status, "clSetKernelArg");

//   // Set the work-item dimensions
//   size_t localWorkSize[2] = {5, 5};
//   size_t globalWorkSize[2] = {imageHeight, imageWidth};

//   // Execute the kernel
//   status = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL,
//   globalWorkSize,
//                                   localWorkSize, 0, NULL, NULL);
//   // CHECK(status, "clEnqueueNDRangeKernel");

//   // Read the output data
//   status = clEnqueueReadImage(commandQueue, outputImageBuffer, CL_TRUE,
//                               (size_t[3]){0, 0, 0},
//                               (size_t[3]){imageWidth, imageHeight, 1}, 0, 0,
//                               outputImage, 0, NULL, NULL);
//   // CHECK(status, "clEnqueueReadBuffer");
// }