#include <immintrin.h>
#include <pthread.h>
#include <stdio.h>

#include <atomic>

std::atomic<uint_fast64_t> total_hits = 0;

struct avx_xorshift128plus_key {
  __m256i part1;
  __m256i part2;
};

static inline void xorshift128plus_onkeys(uint64_t* ps0, uint64_t* ps1) {
  uint64_t s1 = *ps0;
  const uint64_t s0 = *ps1;
  *ps0 = s0;
  s1 ^= s1 << 23;
  *ps1 = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
}

static inline void xorshift128plus_jump_onkeys(uint64_t in1, uint64_t in2,
                                               uint64_t* output1,
                                               uint64_t* output2) {
  constexpr uint64_t JUMP[] = {0x8a5cd789635d2dff, 0x121fd2155c472f96};
  uint64_t s0 = 0;
  uint64_t s1 = 0;
  for (unsigned int i = 0; i < sizeof(JUMP) / sizeof(*JUMP); i++)
    for (int b = 0; b < 64; b++) {
      if (JUMP[i] & 1ULL << b) {
        s0 ^= in1;
        s1 ^= in2;
      }
      xorshift128plus_onkeys(&in1, &in2);
    }
  output1[0] = s0;
  output2[0] = s1;
}

static inline void avx_xorshift128plus_init(uint64_t key1, uint64_t key2,
                                            avx_xorshift128plus_key* key) {
  uint64_t S0[4];
  uint64_t S1[4];
  S0[0] = key1;
  S1[0] = key2;
  xorshift128plus_jump_onkeys(*S0, *S1, S0 + 1, S1 + 1);
  xorshift128plus_jump_onkeys(*(S0 + 1), *(S1 + 1), S0 + 2, S1 + 2);
  xorshift128plus_jump_onkeys(*(S0 + 2), *(S1 + 2), S0 + 3, S1 + 3);
  key->part1 = _mm256_loadu_si256((const __m256i*)S0);
  key->part2 = _mm256_loadu_si256((const __m256i*)S1);
}

static inline __m256i avx_xorshift128plus(avx_xorshift128plus_key* key) {
  __m256i s1 = key->part1;
  const __m256i s0 = key->part2;
  key->part1 = key->part2;
  s1 = _mm256_xor_si256(key->part2, _mm256_slli_epi64(key->part2, 23));
  key->part2 = _mm256_xor_si256(
      _mm256_xor_si256(_mm256_xor_si256(s1, s0), _mm256_srli_epi64(s1, 18)),
      _mm256_srli_epi64(s0, 5));
  return _mm256_add_epi64(key->part2, s0);
}

void* Count(void* args) {
  uint_fast64_t n_tosses = *(uint_fast64_t*)args;
  uint_fast64_t n_hits = 0;

  avx_xorshift128plus_key key;
  avx_xorshift128plus_init(111, 2222222, &key);

  const __m256 KOnes = _mm256_set1_ps(1.0f), kMaxs = _mm256_set1_ps(INT32_MAX);
  __m256 x, y, dist, mask;

  for (uint_fast64_t i = 0; i < n_tosses; i += 8) {
    // Generate 8 random numbers in both x and y.
    x = _mm256_cvtepi32_ps(avx_xorshift128plus(&key));
    y = _mm256_cvtepi32_ps(avx_xorshift128plus(&key));

    // Maps x and y to [-1, 1]
    x = _mm256_div_ps(x, kMaxs);
    y = _mm256_div_ps(y, kMaxs);

    // Compute x * x + y * y
    dist = _mm256_add_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(y, y));

    // Compare dist to 1.0 and increment n_hits if dist is less than 1.0
    mask = _mm256_cmp_ps(dist, KOnes, _CMP_LT_OQ);
    n_hits += _mm_popcnt_u32(_mm256_movemask_ps(mask));
  }

  total_hits += n_hits;

  pthread_exit(nullptr);
}

int main(int argc, char** argv) {
  constexpr int kMaxThreadNum = 16;

  int n_threads = atoi(argv[1]);
  long long n_tosses = atoll(argv[2]);

  pthread_t threads[kMaxThreadNum];
  long long n_tosses_per_thread = n_tosses / n_threads;

  for (int i = 0; i < n_threads; ++i) {
    pthread_create(&threads[i], nullptr, Count, &n_tosses_per_thread);
  }

  for (int i = 0; i < n_threads; ++i) {
    pthread_join(threads[i], nullptr);
  }

  printf("%lf\n", 4. * total_hits / n_tosses);

  return 0;
}
