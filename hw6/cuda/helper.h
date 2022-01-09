#ifndef __HELPER__
#define __HELPER__

#include <stdio.h>
#include <stdlib.h>

// This function reads in a text file and stores it as a char pointer
char *readSource(char *kernelPath);

float *readFilter(const char *filename, int *filterWidth);
#endif