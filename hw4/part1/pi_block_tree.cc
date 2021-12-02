#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

long long int CountPi(long long int tosses) {
  long long int count = 0;
  for (long long int i = 0; i < tosses; ++i) {
    double x = (double)rand() / RAND_MAX;
    double y = (double)rand() / RAND_MAX;
    if (x * x + y * y <= 1) {
      count++;
    }
  }
  return count;
}

int main(int argc, char **argv) {
  // --- DON'T TOUCH ---
  MPI_Init(&argc, &argv);
  double start_time = MPI_Wtime();
  double pi_result;
  long long int tosses = atoi(argv[1]);
  int world_rank, world_size;
  // ---

  // TODO: MPI init
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  srand(time(NULL) * world_rank);
  long long int count = CountPi(tosses / world_size);

  // TODO: binary tree redunction
  unsigned offset = 1;
  while (offset < world_size) {
    if (world_rank % (offset << 1) == offset) {
      MPI_Send(&count, 1, MPI_LONG_LONG_INT, world_rank - offset, 0,
               MPI_COMM_WORLD);
      break;
    } else if (world_rank % (offset << 1) == 0) {
      long long int total;
      MPI_Recv(&total, 1, MPI_LONG_LONG_INT, world_rank + offset, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      count += total;
    }
    offset <<= 1;
  }

  if (world_rank == 0) {
    // TODO: PI result
    pi_result = (double)count / (double)tosses * 4;

    // --- DON'T TOUCH ---
    double end_time = MPI_Wtime();
    printf("%lf\n", pi_result);
    printf("MPI running time: %lf Seconds\n", end_time - start_time);
    // ---
  }

  MPI_Finalize();
  return 0;
}
