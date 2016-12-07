#include "cudacomplex.h"

#define conjf(c) ((c).conjugate())
#define crealf(c) ((c).re())
#define cimagf(c) ((c).im())
#define cabsf(c) ((c).abs())
#define norm(c) ((c).norm())
#define cZero singlecomplex(0.0F, 0.0F)
#define cOne singlecomplex(1.0F, 0.0F)

/* Below were for using thrust/complex */
// #define singlecomplex thrust::complex<float>
// #define conjf(c) (conj(c))
// #define crealf(c) ((c).real())
// #define cimagf(c) ((c).imag())
// #define cabsf(c) (abs(c))
// #define norm(c) (norm(c))
// #define cZero singlecomplex(0.0F, 0.0F)
// #define cOne singlecomplex(1.0F, 0.0F)
// #define singlecomplex singlecomplex

#ifndef maxA
#define maxA 4
#endif

#ifndef INTERLEAVE
#define INTERLEAVE 256
#endif

#ifndef BATCH_SIZE
#define BATCH_SIZE 4096
#endif
