#include <stdlib.h>
#include <complex.h>
#include <inttypes.h>

#define MIN(a,b) ((a < b)? a : b)
#define MAX(a,b) ((a > b)? a : b)
#define ABS(a) ((a > 0) ? a : -a)
#define EPSILON 1e-4
#define complexEqual(z1, z2) (ABS(__real__(z1) - __real__(z2)) < EPSILON && ABS(__imag__(z1) - __imag__(z2)) < EPSILON)

float _Complex* generateChessboard (int width, int height, int boardBlockWidth, int boardBlockHeight);
float _Complex* generateRandomMatrix (int width, int height);
char* diffMatrix (float _Complex* m1, float _Complex* m2, int width, int height);
void printMatrix (float _Complex *m, int width, int height);
void printMatrixPrecise (float _Complex *m, int width, int height);

float _Complex randComplex (int max);

inline uint64_t rdtsc() {
    uint32_t lo, hi;
    __asm__ __volatile__ (
      "xorl %%eax, %%eax\n"
      "cpuid\n"
      "rdtsc\n"
      : "=a" (lo), "=d" (hi)
      :
      : "%ebx", "%ecx");
    return (uint64_t)hi << 32 | lo;
}
