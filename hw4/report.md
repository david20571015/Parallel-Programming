# Parallel Programming HW4

###### tags: `Parallel Programming`

## Q1

### Q1-1

> How do you control the number of MPI processes on each node?

* Indicate nodes by `--host` option to mpirun.

```bash
mpirun --host pp2,pp3,pp3,pp4,pp4,pp4 mpi_hello
```

![Q1-1-1](https://i.imgur.com/9sSUSSM.png)

* Indicate nodes and the process number for each node by `--host` option to mpirun.

```bash
mpirun --host pp2,pp3:2,pp4:3 mpi_hello
```

![Q1-1-2](https://i.imgur.com/Vwn1QfD.png)

* Indicate `slots` in the hostfile.

```bash
mpirun --hostfile hosts mpi_hello
```

![Q1-1-3](https://i.imgur.com/rBc1xQE.png)

**reference:**

* [mpirun scheduling](https://www.open-mpi.org/faq/?category=running#mpirun-scheduling)
* [mpirun options](https://www.open-mpi.org/doc/v4.0/man1/mpirun.1.php#sect6)

### Q1-2

> Which functions do you use for retrieving the rank of an MPI process and the total number of processes?

* Use `MPI_Comm_rank` for retrieving the rank of an MPI process

```c
int world_rank;
MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
```

* Use `MPI_Comm_size` for retrieving the total number of processes.

```c
int world_size;
MPI_Comm_size(MPI_COMM_WORLD, &world_size);
```

**reference:**

* [MPI_Comm_rank](https://www.open-mpi.org/doc/v3.1/man3/MPI_Comm_rank.3.php)
* [MPI_Comm_size](https://www.open-mpi.org/doc/v3.1/man3/MPI_Comm_size.3.php)

## Q2

### Q2-1

> Why `MPI_Send` and `MPI_Recv` are called “blocking” communication?

These function do not return until the communication finished. Therefore, the program will be blocked while communicating to other processes.

### Q2-2

> Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.

| MPI processes | 2         | 4        | 8        | 12       | 16       |
| ------------- | --------- | -------- | -------- | -------- | -------- |
| Time cost     | 16.193483 | 8.342283 | 4.143358 | 2.815187 | 2.098945 |
| Speedup       | 1.00x     | 1.94x    | 3.91x    | 5.75x    | 7.71x    |

![Q2-2](https://i.imgur.com/xm9yIMg.png)

## Q3

### Q3-1

> Measure the performance (execution time) of the code for 2, 4, 8, 16 MPI processes and plot it.

| MPI processes | 2         | 4        | 8        | 16       |
| ------------- | --------- | -------- | -------- | -------- |
| Time cost     | 15.834398 | 8.397105 | 4.132318 | 2.095542 |
| Speedup       | 1.00x     | 1.88x    | 3.83x    | 7.56x    |

![Q3-1](https://i.imgur.com/tbEylzi.png)

### Q3-2

> How does the performance of binary tree reduction compare to the performance of linear reduction?

The performance of two methods are almost the same. While using **N** MPI processes, both methods need **N - 1** additions and **N - 1** communications.

### Q3-3

> Increasing the number of processes, which approach (linear/tree) is going to perform better? Why? Think about the number of messages and their costs.

From Q2-2 and Q3-1, the speedup is roughly proportional to the number of processes, and thus we can infer that the **computation** is the bottleneck in this program. These two approaches perform almost the same bacause both of them need same times of additions.

If the communications is the bottleneck, tree might be the better approach. Master process needs to receive data from **N - 1** processes by linear approach, but **log2(N)** processes by tree approach.

## Q4

### Q4-1

> Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.

| MPI processes | 2         | 4        | 8        | 12       | 16       |
| ------------- | --------- | -------- | -------- | -------- | -------- |
| time(sec)     | 15.827162 | 8.661146 | 4.111082 | 2.752440 | 2.058296 |

![Q4-1](https://i.imgur.com/er6wyEy.png)

### Q4-2

> What are the MPI functions for non-blocking communication?

There are many MPI functions for non-blocking communication, for example

| MPI Function | Description                            |
| ------------ | -------------------------------------- |
| MPI_Isend    | Begins a nonblocking send.             |
| MPI_Irecv    | Begins a nonblocking receive.          |
| MPI_Wait     | Waits for an MPI request to complete.  |
| MPI_Test     | Tests for the completion of a request. |

Those functions with prefix "MPI_I" are usually for non-blocking communication.

**reference:**

* [MPI functions](https://www.mpich.org/static/docs/v3.1.x/www3/)

### Q4-3

> How the performance of non-blocking communication compares to the performance of blocking communication?

| MPI processes | 2        | 4        | 8        | 12       | 16       |
| ------------- | -------- | -------- | -------- | -------- | -------- |
| Speedup       | 1.023145 | 0.963184 | 1.007851 | 1.022797 | 1.019749 |

Non-blocking communication has a slight speedup compares to blocking communication. Because we do nothing between MPI_Irecv and MPI_Waitall, this non-blocking approach works like blocking approach.

## Q5

### Q5-1

> Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.

| MPI processes | 2         | 4        | 8        | 12       | 16       |
| ------------- | --------- | -------- | -------- | -------- | -------- |
| time(sec)     | 15.869996 | 8.217893 | 4.099428 | 2.850659 | 2.209958 |

![Q5-1](https://i.imgur.com/EzbxWwR.png)

## Q6

### Q6-1

> Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.

| MPI processes | 2         | 4        | 8        | 12       | 16       |
| ------------- | --------- | -------- | -------- | -------- | -------- |
| time(sec)     | 15.877728 | 8.304110 | 4.293003 | 2.772876 | 2.289611 |

![Q6-1](https://i.imgur.com/qwHVrIs.png)

## Q7

### Q7-1

> Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.

| MPI processes | 2         | 4        | 8        | 12       | 16       |
| ------------- | --------- | -------- | -------- | -------- | -------- |
| time(sec)     | 15.849764 | 8.157315 | 4.160088 | 2.770919 | 2.101583 |

![Q7-1](https://i.imgur.com/1pEqpGV.png)

### Q7-2

> Which approach gives the best performance among the **[1.2.1](https://nycu-sslab.github.io/PP-f21/HW4/#121--mpi-blocking-communication--linear-reduction-algorithm)**-**[1.2.6](https://nycu-sslab.github.io/PP-f21/HW4/#126-mpi-windows-and-one-sided-communication--linear-reduction-algorithm)** cases? What is the reason for that?

| MPI processes          |     2     |    4     |    8     |    12    |    16    |
| ---------------------- | :-------: | :------: | :------: | :------: | :------: |
| Blocking & Linear      | 16.193483 | 8.342283 | 4.143358 | 2.815187 | 2.098945 |
| Blocking & Binary Tree | 15.834398 | 8.397105 | 4.132318 |    -     | 2.095542 |
| Non-Blocking & Linear  | 15.827162 | 8.661146 | 4.111082 | 2.752440 | 2.058296 |
| MPI_Gather             | 15.869996 | 8.217893 | 4.099428 | 2.850659 | 2.209958 |
| MPI_Reduce             | 15.877728 | 8.304110 | 4.293003 | 2.772876 | 2.289611 |
| MPI Windows            | 15.849764 | 8.157315 | 4.160088 | 2.770919 | 2.101583 |

![Q7-2](https://i.imgur.com/5Q4WgEw.png)

As Q3-3, there is no significant difference in each approach because of heavy **computation**.  

## Q8

### Q8-1

> Plot ping-pong time in function of the message size for cases 1 and 2, respectively.

![case 1](https://i.imgur.com/mylnhIe.png)

![case 2](https://i.imgur.com/5Wrzvj9.png)

### Q8-2

> Calculate the bandwidth and latency for cases 1 and 2, respectively.

With regression line *y = mx + b*

* bandwidth = 1 / m
* latency = b

| Case              | 1      | 2       |
| ----------------- | ------ | ------- |
| bandwidth (bytes) | 5e9    | 1.111e8 |
| latency (sec)     | 0.0003 | 0.0009  |

## Q9

### Q9-1

> Describe what approach(es) were used in your MPI matrix multiplication for each data set.

For matrix A, I divide it into *world_size* blocks by its row and send each block to each process. For matrix B, I broadcase the whole matrix to other processes. Then, each process computes **A(blocked) * B = C(blocked)** and send the rasult to rank 0 (master process). After rank 0 (master process) receive all blocks of matrix C, it combines the matrix and prints the whole matrix to stdout.
