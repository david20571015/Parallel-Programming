# Parallel Programming HW2

## Q1

> In your write-up, produce a graph of **speedup compared to the reference sequential implementation** as a function of the number of threads used **FOR VIEW 1**. Is speedup linear in the number of threads used? In your writeup hypothesize why this is (or is not) the case?

![line chart](https://i.imgur.com/4NKRdqw.png)

Obviously, the speedup **is not** linear in the number of threads used for view 1, but it is linear for view 2. We know that the total time cost of a multi-thread is rouhgly equal to the largest time cost of the threads. To reveal the reason of non-linear speedup, we can observe the image of both view and estimate the computational cost by observe the bright area of the image.

* View 1
When we use 2 or 4 threads, the computational cost equally split to every thread. In contrast with using 3 threads, the 2nd thread's computational cost might be even greater than any thread while using only 2 threads. Therefore, the speedup drops while using 3 threads.
![View 1, 2 threads](https://i.imgur.com/4s7RJBd.png)
![View 1, 3 threads](https://i.imgur.com/hkZIygQ.png)
![View 1, 4 threads](https://i.imgur.com/VdqXisV.png)

* View 2
For view 2, the 1st thread always compute the most number of value contained in the Mandelbrot set. But as we use more threads, the 1st thread's computational cost decreases. Thus, the speedup is roughly linear in the number of threads used.
![View 2, 2 threads](https://i.imgur.com/2X1TBvo.png)
![View 2, 3 threads](https://i.imgur.com/K58vyyV.png)
![View 2, 4 threads](https://i.imgur.com/wb08lyb.png)

## Q2

> How do your measurements explain the speedup graph you previously created?

* View 1

| time cost | 1st thread | 2nd thread | 3rd thread | 4th thread | bottleneck |
| --------- | :--------: | :--------: | :--------: | :--------: | :--------: |
| 2 threads | 239.1594   | 240.019    | --         | --         | 240.019    |  
| 3 threads | 94.0144    | 275.145    | 94.4036    | --         | 275.145    |
| 4 threads | 45.3935    | 193.1046   | 193.8174   | 45.8033    | 193.8174   |

The largest bottleneck is 275.145 ms while using **3 threads**.

* View 2

| time cost | 1st thread | 2nd thread | 3rd thread | 4th thread | bottleneck |
| --------- | :--------: | :--------: | :--------: | :--------: | :--------: |
| 2 threads | 172.5674   | 122.7282   | --         | --         | 172.5674   |  
| 3 threads | 133.8058   | 86.6178    | 79.5375    | --         | 133.8058   |
| 4 threads | 110.9054   | 65.9612    | 64.8894    | 60.9906    | 110.9054   |

The bottleneck decrease while we using more threads.

## Q3

> In your write-up, describe your approach to parallelization and report the final 4-thread speedup obtained.

From the result of [Q1](https://hackmd.io/gnODwIojQuiYGoRfnRhE_g#Q1) and [Q2](https://hackmd.io/gnODwIojQuiYGoRfnRhE_g#Q2), we can conclude that the computational cost is not equally split to each thread. Thus, I modify my program from

```cpp=
void workerThreadStart(WorkerArgs *const args) {
  int numRows = args->height / args->numThreads;
  int startRow = args->threadId * numRows;
  
  mandelbrotSerial(args->x0, args->y0, args->x1, args->y1, args->width, 
                   args->height, startRow, numRows, args->maxIterations,
                   args->output);
}
```

to

```cpp=
void workerThreadStart(WorkerArgs *const args) {
  for (int i = args->threadId; i < args->height; i += args->numThreads) {
    mandelbrotSerial(args->x0, args->y0, args->x1, args->y1, args->width,
                     args->height, i, 1, args->maxIterations, args->output);
  }
}
```

which mod those rows' indexes by `numThreads` then group congruent modulo indexes together. Thus, every thread takes turns to handle each row and prevent some thread from heavier computing.

## Q4

> Now run your improved code with eight threads. Is performance noticeably greater than when running with four threads? Why or why not?

<!-- ![](https://i.imgur.com/KW81Phw.png) -->

|  Speedup  | view 1 | view 2 |
| :-------: | :----: | :----: |
| 4 threads | 3.80   | 3.79   |
| 8 threads | 3.71   | 3.79   |

The performace of using 8 threads is not noticeablly greater but even worse than using 4 threads since we execute this program on a 4-thread server. If we use threads that more than a server provided, it will lead to redundant thread context switch and decreases the speedup.
