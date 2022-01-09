#include "helper.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// This function reads in a text file and stores it as a char pointer
char *readSource(char *kernelPath) {
  int status;
  FILE *fp;
  char *source;
  long int size;

  printf("Program file is: %s\n", kernelPath);

  fp = fopen(kernelPath, "rb");
  if (!fp) {
    printf("Could not open kernel file\n");
    exit(-1);
  }
  status = fseek(fp, 0, SEEK_END);
  if (status != 0) {
    printf("Error seeking to end of file\n");
    exit(-1);
  }
  size = ftell(fp);
  if (size < 0) {
    printf("Error getting file position\n");
    exit(-1);
  }

  rewind(fp);

  source = (char *)malloc(size + 1);

  int i;
  for (i = 0; i < size + 1; i++) {
    source[i] = '\0';
  }

  if (source == NULL) {
    printf("Error allocating space for the kernel source\n");
    exit(-1);
  }

  fread(source, 1, size, fp);
  source[size] = '\0';

  return source;
}

float *readFilter(const char *filename, int *filterWidth) {
  printf("Reading filter data from %s\n", filename);

  FILE *fp = fopen(filename, "r");
  if (!fp) {
    printf("Could not open filter file\n");
    exit(-1);
  }

  fscanf(fp, "%d", filterWidth);

  float *filter = (float *)malloc(*filterWidth * *filterWidth * sizeof(int));

  float tmp;
  for (int i = 0; i < *filterWidth * *filterWidth; i++) {
    fscanf(fp, "%f", &tmp);
    filter[i] = tmp;
  }

  printf("Filter width: %d\n", *filterWidth);

  fclose(fp);
  return filter;
}
