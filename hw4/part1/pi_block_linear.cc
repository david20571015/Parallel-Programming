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

  // TODO: init MPI
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  srand(time(NULL) * world_rank);
  long long int count = CountPi(tosses / world_size);

  if (world_rank > 0) {
    // TODO: handle workers
    int dest = 0;
    MPI_Send(&count, 1, MPI_LONG_LONG_INT, dest, 0, MPI_COMM_WORLD);
  } else if (world_rank == 0) {
    // TODO: master
    long long int total;
    for (int src = 1; src < world_size; ++src) {
      MPI_Recv(&total, 1, MPI_LONG_LONG_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      count += total;
    }
  }

  if (world_rank == 0) {
    // TODO: process PI result
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
