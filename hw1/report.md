# Parallel Programming HW1

## Q1

### Q1-1

> Run `./myexp -s 10000` and sweep the vector width from 2, 4, 8, to 16. Record the resulting vector utilization. You can do this by changing the `#define VECTOR_WIDTH` value in `def.h` . Does the vector utilization increase, decrease or stay the same as `VECTOR_WIDTH` changes? Why?

* VECTOR_WIDTH = 2

![2](https://i.imgur.com/iKWwzNp.png)

* VECTOR_WIDTH = 4

![4](https://i.imgur.com/4HznSyT.png)

* VECTOR_WIDTH = 8

![8](https://i.imgur.com/XpDl7vN.png)

* VECTOR_WIDTH = 16

![16](https://i.imgur.com/x3vFWpG.png)

| VECTOR_WIDTH | Vector Utilization |
|:------------:|:------------------:|
| 2            | 91.3%              |
| 4            | 87.7%              |
| 8            | 85.9%              |
| 16           | 85.1%              |

The vector utilization decreases as `VECTOR_WIDTH` increasing.
1. While `VECTOR_WIDTH` increasing, it is more likely that higher exponential and lower exponential in the same vector. Thus decreases the vector utilization.
3. If the data size (N) is not divisible by `VECTOR_WIDTH`,  `N % VECTOR_WIDTH` element in the last vector will always be masked. Hence lead to low vector utilization. However this effect will become smaller when N becomes larger.

## Q2

### Q2-1

> Fix the code to make sure it uses aligned moves for the best performance.

Replace
[Compiler Explorer](https://godbolt.org/z/rah9Yv9z6)

```c
  a = (float *)__builtin_assume_aligned(a, 16);
  b = (float *)__builtin_assume_aligned(b, 16);
  c = (float *)__builtin_assume_aligned(c, 16);
```

by
[Compiler Explorer](https://godbolt.org/z/Yxza7b3rY)

```c
  a = (float *)__builtin_assume_aligned(a, 32);
  b = (float *)__builtin_assume_aligned(b, 32);
  c = (float *)__builtin_assume_aligned(c, 32);
```

AVX2 instruction operate with 256-bit wide YMM registers. So we must set the align size to at least *32 bytes* (256 bits) for `__builtin_assume_aligned` .

**reference:**

* [Intrinsics for Intel® Advanced Vector Extensions 2 (Intel® AVX2)](https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-avx2.html#:~:text=Intel%C2%AE%20AVX2%20instructions%20promote%20the%20vast%20majority%20of%20128-bit%20integer%20SIMD%20instruction%20sets%20to%20operate%20with%20256-bit%20wide%20YMM%20registers.)
* [Other Builtins (Using the GNU Compiler Collection (GCC))](https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html#:~:text=Built-in%20Function%3A%20void%20*%20__builtin_assume_aligned)

### Q2-2

> What speedup does the vectorized code achieve over the unvectorized code? What additional speedup does using -mavx2 give (AVX2=1 in the Makefile)? You may wish to run this experiment several times and take median elapsed times; you can report answers to the nearest 100% (e.g., 2×, 3×, etc).

* `make clean && make && ./test_auto_vectorize -t 1` [Compiler Explorer](https://godbolt.org/z/dnThvnrcd)

![case 1](https://i.imgur.com/5b91GJf.png)

* `make clean && make VECTORIZE=1 && ./test_auto_vectorize -t 1` [Compiler Explorer](https://godbolt.org/z/bs1nzKMva)

![case 2](https://i.imgur.com/sThK5mJ.png)

* `make clean && make VECTORIZE=1 AVX2=1 && ./test_auto_vectorize -t 1` [Compiler Explorer](https://godbolt.org/z/x96ssPhn5)

![case 3](https://i.imgur.com/Z7jUGNf.png)

|      Flags      |  Time Cost  | Speed Up |
|:---------------:|:-----------:|:--------:|
| (empty)         | 8.25204 sec | 1x       |
| VECTORIZE       | 2.61875 sec | 3.15x    |
| VECTORIZE, AVX2 | 1.39504 sec | 5.92x    |

> What can you infer about the bit width of the default vector registers on the PP machines? What about the bit width of the AVX2 vector registers?

We can infer that the bit width of the default vector registers on the PP machines is 32 bits * 3  = 96 bits, because case 2 is about 3 times as fast as case 1. Also, the AVX2 vector width is 32 bits * 6  = 192 bits.

However, we know the AVX2 vector width is *256 bits* from its official document. Thus, the default vector width on the PP machines is 256 bits / 2 = *128 bits*.

From the assembly code of these three cases, we can observe that there is overhead in case 2 and case 3 to check if the program is vectorizable. It increases the execution time and cause the error of our result.

**reference:**

* [Intrinsics for Intel® Advanced Vector Extensions 2 (Intel® AVX2)](https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-avx2.html#:~:text=Intel%C2%AE%20AVX2%20instructions%20promote%20the%20vast%20majority%20of%20128-bit%20integer%20SIMD%20instruction%20sets%20to%20operate%20with%20256-bit%20wide%20YMM%20registers.)

### Q2-3

> Provide a theory for why the compiler is generating dramatically different assembly.

[Compiler Explorer](https://godbolt.org/z/1EvGd8aYG)

```c
// origin
c[j] = a[j];
if (b[j] > a[j])
    c[j] = b[j];
```

The abstract syntax tree of origin code is

* Assign
* Condition
  + Assign
  + Null

this is not a parallelizable program. Because it assign `c[j]` to `a[j]` , compare `b[j]` and `a[j]` , then assign `c[j]` to `b[j]` if `b[j] > a[j]` .

[Compiler Explorer](https://godbolt.org/z/YbMEYWYPj)

```c
// patched
if (b[j] > a[j]) c[j] = b[j];
else c[j] = a[j];
```

But the abstract syntax tree of patched code is

* Condition
  + Assign
  + Assign

compiler will vectorize it using `maxps` from SIMD, hence it will be able to accelerate by AVX2.
