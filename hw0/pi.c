#include <stdio.h>
#include <stdlib.h>

int main(void) {
  long long int number_in_circle = 0;
  long long int number_of_tosses = 1e9;

  for (long long int toss = 0; toss < number_of_tosses; ++toss) {
    double x = ((double)rand()) / RAND_MAX * 2. - 1.;
    double y = ((double)rand()) / RAND_MAX * 2. - 1.;

    double distance_squared = x * x + y * y;
    if (distance_squared <= 1.) {
      ++number_in_circle;
    }
  }

  double pi_estimate = 4. * number_in_circle / ((double)number_of_tosses);

  printf("pi_estimate = %f\n", pi_estimate);

  return 0;
}