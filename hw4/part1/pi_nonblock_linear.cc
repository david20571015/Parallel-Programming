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

  long long int *recv;

  if (world_rank > 0) {
    // TODO: MPI workers
    MPI_Request request;
    int dest = 0;
    MPI_Isend(&count, 1, MPI_LONG_LONG_INT, dest, 0, MPI_COMM_WORLD, &request);
  } else if (world_rank == 0) {
    // TODO: non-blocking MPI communication.
    // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
    recv = (long long int *)malloc(sizeof(long long int) * world_size);
    MPI_Request *requests =
        (MPI_Request *)malloc(sizeof(MPI_Request) * world_size);

    for (int src = 1; src < world_size; ++src) {
      MPI_Irecv(recv + src, 1, MPI_LONG_LONG_INT, src, 0, MPI_COMM_WORLD,
                requests + src - 1);
    }

    MPI_Waitall(world_size - 1, requests, MPI_STATUSES_IGNORE);
    free(requests);
  }

  if (world_rank == 0) {
    // TODO: PI result
    for (int i = 1; i < world_size; ++i) {
      count += recv[i];
    }
    free(recv);
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
