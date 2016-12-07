#include "cudaBatchSVT_macros.h"

/*Convience macro for interleaved access*/
#define access_m(a,i,j) a[i][(j)*BATCH_SIZE]
#define access_a(a,i) a[(i)*BATCH_SIZE]

/*Convenient macro for const __restrict__ array interleaved access*/
#define res_acc_m(a,i,j) a[(i) * maxA * BATCH_SIZE + (j) * BATCH_SIZE]
#define res_acc_a(a,i) a[(i)*BATCH_SIZE]

extern "C" __device__ void CSVD(singlecomplex a[][maxA * BATCH_SIZE], int m, int n,
		float s[], singlecomplex u[][maxA * BATCH_SIZE], singlecomplex v[][maxA * BATCH_SIZE], float *work);

extern "C" __global__ void batchSVD(singlecomplex *a, int m, int n,
    float *s, singlecomplex *u, singlecomplex *v, float * work);

extern "C" __device__ void htridi ( int n, singlecomplex a[][maxA * BATCH_SIZE], float d[maxA * BATCH_SIZE], 
  float e[maxA], float tau[2][maxA * BATCH_SIZE]);

#ifdef __CUDACC_RTC__
#define batchCHQL deviceCHQLFName
#endif

extern "C" __global__ void batchCHQL(singlecomplex *a, int n, float *s, 
  singlecomplex *u, float *work);

#ifndef __CUDACC_RTC__
void nvrtc_batchCHQL(singlecomplex *a, int m, int n, float *s, 
  singlecomplex *u, float *work, int batch_size);

void batch_clascl_diag_full(int m, int n, int batch_size, const float *dD, singlecomplex *dA, int ldda);
void batch_clascl_diag_thres_full(int m, int n, int batch_size, const float *dD, singlecomplex *dA, int ldda, float thres);
#endif
