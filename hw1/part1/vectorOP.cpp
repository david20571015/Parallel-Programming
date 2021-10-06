#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N) {
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH) {
    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll);  // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll);  // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative);  //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative);  // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative);  //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N) {
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //

  // constant
  __pp_vec_int allZeros = _pp_vset_int(0), allOnes = _pp_vset_int(1);
  __pp_vec_float upperBound = _pp_vset_float(9.999999f);

  __pp_vec_int y;
  __pp_vec_float x, result;
  __pp_mask maskAll, maskZeroExp, maskNotZeroExp;

  for (int i = 0; i < N; i += VECTOR_WIDTH) {
    // mask for all valiable elements
    int maskSize = min(N - i, VECTOR_WIDTH);
    maskAll = _pp_init_ones(maskSize);

    // x = values, y = exponents
    _pp_vload_float(x, values + i, maskAll);
    _pp_vload_int(y, exponents + i, maskAll);

    // result = 1.
    _pp_vset_float(result, 1.f, maskAll);

    // update maskZeroExp and maskNotZeroExp
    _pp_veq_int(maskZeroExp, y, allZeros, maskAll);
    maskZeroExp = _pp_mask_and(maskZeroExp, maskAll);
    maskNotZeroExp = _pp_mask_not(maskZeroExp);
    maskNotZeroExp = _pp_mask_and(maskNotZeroExp, maskAll);

    while (_pp_cntbits(maskZeroExp) < maskSize) {
      // result *= x
      _pp_vmult_float(result, result, x, maskNotZeroExp);
      // y--
      _pp_vsub_int(y, y, allOnes, maskNotZeroExp);

      // update maskZeroExp and maskNotZeroExp
      _pp_veq_int(maskZeroExp, y, allZeros, maskAll);
      maskNotZeroExp = _pp_mask_not(maskZeroExp);
      maskNotZeroExp = _pp_mask_and(maskNotZeroExp, maskAll);
    }

    // result = max(result, 9.999999f)
    __pp_mask maskClamp;
    _pp_vgt_float(maskClamp, result, upperBound, maskAll);
    _pp_vmove_float(result, upperBound, maskClamp);

    // output = result
    _pp_vstore_float(output + i, result, maskAll);
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N) {
  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  __pp_mask maskAll = _pp_init_ones();

  int currentSize;
  float sum = 0.0;
  __pp_vec_float compressed;

  for (int i = 0; i < N; i += VECTOR_WIDTH) {
    currentSize = VECTOR_WIDTH;
    _pp_vload_float(compressed, values + i, maskAll);

    while (currentSize > 1) {
      // [0 1 2 3] -> [0+1 0+1 2+3 2+3]
      _pp_hadd_float(compressed, compressed);
      // [0+1 0+1 2+3 2+3] -> [0+1 2+3 0+1 2+3]
      _pp_interleave_float(compressed, compressed);

      currentSize /= 2;
    }
    sum += compressed.value[0];
  }

  return sum;
}