__kernel void convolution(int filterWidth, __constant float *filter,
                          int imageHeight, int imageWidth,
                          __global const float *inputImage,
                          __global float *outputImage) {
  const int i = get_global_id(0);
  const int j = get_global_id(1);

  int halffilterSize = filterWidth / 2;
  float sum = 0.0f;
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

  outputImage[i * imageWidth + j] = sum;
}

// __kernel void convolution(int filterWidth, __constant float *filter,
//                           int imageHeight, int imageWidth,
//                           __read_only image2d_t inputImage,
//                           __write_only image2d_t outputImage) {
//   const int i = get_global_id(0);
//   const int j = get_global_id(1);

//   const sampler_t smp =
//       CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

//   int halffilterSize = filterWidth / 2;
//   float sum;
//   int filterIdx = -1;

//   for (int k = -halffilterSize; k <= halffilterSize; k++) {
//     if (i + k < 0 || i + k >= imageHeight) continue;
//     for (int l = -halffilterSize; l <= halffilterSize; l++) {
//       if (j + l < 0 || j + l >= imageWidth) continue;

//       if (filter[++filterIdx] == 0) continue;
//       sum +=
//           filter[filterIdx] * read_imagef(inputImage, (int2)(j + l, i +
//           k)).x;
//     }
//   }
//   write_imagef(outputImage, (int2)(j, i), (float4)(sum, sum, sum, sum));
// }

// __kernel void convolution(int filterWidth, __constant float *filter,
//                           int imageHeight, int imageWidth,
//                           const __global float *inputImage,
//                           __global float4 *outputImage) {
//   const int i = get_global_id(0);
//   const int j = get_global_id(1) << 2;

//   int halffilterSize = filterWidth / 2;
//   __local float4 sum;
//   sum = (float4){0.0f, 0.0f, 0.0f, 0.0f};

//   for (int k = -halffilterSize; k <= halffilterSize; k++) {
//     if (i + k < 0 || i + k >= imageHeight) continue;
//     for (int l = -halffilterSize; l <= halffilterSize; l++) {
//       if (j + l < 0 || j + l >= imageWidth) continue;

//       int imageIdx = (i + k) * imageWidth + j + l;
//       float4 pixel =
//           (float4){inputImage[imageIdx], inputImage[imageIdx + 1],
//                    inputImage[imageIdx + 2], inputImage[imageIdx + 3]};
//       sum += pixel *
//              filter[(k + halffilterSize) * filterWidth + l + halffilterSize];
//     }
//   }

//   outputImage[(i * imageWidth + j) >> 2] = sum;
// }
