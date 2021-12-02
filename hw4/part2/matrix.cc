#include <mpi.h>
#include <stdlib.h>

#include <cstdio>

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr,
                        int **b_mat_ptr) {

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // double start_time = MPI_Wtime();

  char *buffer, *p;

  if (rank == 0) {
    buffer = (char *)malloc(sizeof(char) * 1 << 30);
    p = buffer;
    int len = fread(buffer, sizeof(char), 1 << 30, stdin);

    *n_ptr = *m_ptr = *l_ptr = 0;

    for (; '0' <= *p && *p <= '9'; ++p) {
      *n_ptr = *n_ptr * 10 + *p - '0';
    }
    ++p;

    for (; '0' <= *p && *p <= '9'; ++p) {
      *m_ptr = *m_ptr * 10 + *p - '0';
    }
    ++p;

    for (; '0' <= *p && *p <= '9'; ++p) {
      *l_ptr = *l_ptr * 10 + *p - '0';
    }
    ++p;
  }

  MPI_Bcast(n_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(m_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(l_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

  *a_mat_ptr = (int *)malloc(sizeof(int) * (*n_ptr) * (*m_ptr));
  *b_mat_ptr = (int *)malloc(sizeof(int) * (*m_ptr) * (*l_ptr));

  if (rank == 0) {
    for (int i = 0; i < (*n_ptr) * (*m_ptr); ++i) {
      (*a_mat_ptr)[i] = 0;
    }

    for (int count = 0; count < (*n_ptr) * (*m_ptr); ++p) {
      if ('0' <= *p && *p <= '9') {
        (*a_mat_ptr)[count] = (*a_mat_ptr)[count] * 10 + *p - '0';
      } else if (*p == ' ') {
        ++count;
      }
    }

    for (int i = 0; i < (*m_ptr) * (*l_ptr); ++i) {
      (*b_mat_ptr)[i] = 0;
    }

    for (int count = 0; count < (*m_ptr) * (*l_ptr); ++p) {
      if ('0' <= *p && *p <= '9') {
        (*b_mat_ptr)[count] = (*b_mat_ptr)[count] * 10 + *p - '0';
      } else if (*p == ' ') {
        ++count;
      }
    }

    free(buffer);
  }

  // double end_time = MPI_Wtime();
  // printf("MPI input time: %lf Seconds in rank %d\n", end_time - start_time,
  //        rank);

  // start_time = MPI_Wtime();

  constexpr int MAX_PROCESS_NUM = 9;
  int start[MAX_PROCESS_NUM];
  int end[MAX_PROCESS_NUM];

  int max_used_process_num = *n_ptr < size ? *n_ptr : size;
  const float BLOCK_SIZE = 1.0 * *n_ptr / max_used_process_num;
  for (int i = 0; i < max_used_process_num - 1; ++i) {
    start[i] = BLOCK_SIZE * i;
    end[i] = BLOCK_SIZE * (i + 1);
  }

  start[max_used_process_num - 1] =
      max_used_process_num == 1 ? 0 : end[max_used_process_num - 2];
  end[max_used_process_num - 1] = *n_ptr;

  int sendcounts[MAX_PROCESS_NUM], displs[MAX_PROCESS_NUM];
  for (int i = 0; i < max_used_process_num; ++i) {
    sendcounts[i] = (end[i] - start[i]) * (*m_ptr);
    displs[i] = start[i] * (*m_ptr);
  }

  MPI_Scatterv(*a_mat_ptr, sendcounts, displs, MPI_INT, *a_mat_ptr,
               sendcounts[rank], MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Bcast(*b_mat_ptr, (*m_ptr) * (*l_ptr), MPI_INT, 0, MPI_COMM_WORLD);
  // end_time = MPI_Wtime();

  // printf("MPI bcast time: %lf Seconds in rank %d\n", end_time - start_time,
  //        rank);
}

void matrix_multiply(const int n, const int m, const int l, const int *a_mat,
                     const int *b_mat) {
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  constexpr int MAX_PROCESS_NUM = 9;
  int start[MAX_PROCESS_NUM];
  int end[MAX_PROCESS_NUM];

  int max_used_process_num = n < size ? n : size;
  const float BLOCK_SIZE = 1.0 * n / max_used_process_num;
  for (int i = 0; i < max_used_process_num - 1; ++i) {
    start[i] = BLOCK_SIZE * i;
    end[i] = BLOCK_SIZE * (i + 1);
  }

  start[max_used_process_num - 1] =
      max_used_process_num == 1 ? 0 : end[max_used_process_num - 2];
  end[max_used_process_num - 1] = n;

  // double start_time = MPI_Wtime();

  int *c_mat = (int *)malloc(sizeof(int) * n * l);

  if (rank < max_used_process_num) {
    for (int i = start[rank]; i < end[rank]; ++i) {
      for (int k = 0; k < l; ++k) {
        int sum = 0;
        for (int j = 0; j < m; ++j) {
          sum += a_mat[i * m + j] * b_mat[j * l + k];
        }
        c_mat[i * l + k] = sum;
      }
    }
  }

  // double end_time = MPI_Wtime();
  // printf("MPI calculate time: %lf Seconds in rank %d\n", end_time - start_time,
  //        rank);

  // start_time = MPI_Wtime();

  int sendcounts[MAX_PROCESS_NUM], displs[MAX_PROCESS_NUM];
  for (int i = 0; i < max_used_process_num; ++i) {
    sendcounts[i] = (end[i] - start[i]) * l;
    displs[i] = start[i] * l;
  }

  MPI_Gatherv(c_mat, sendcounts[rank], MPI_INT, c_mat, sendcounts, displs,
              MPI_INT, 0, MPI_COMM_WORLD);

  // end_time = MPI_Wtime();
  // printf("MPI collect time: %lf Seconds in rank %d\n", end_time - start_time,
  //        rank);

  if (rank == 0) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < l; ++j) {
        printf("%d ", c_mat[i * l + j]);
      }
      printf("\n");
    }
  }

  free(c_mat);
}

void destruct_matrices(int *a_mat, int *b_mat) {
  free(a_mat);
  free(b_mat);
}
