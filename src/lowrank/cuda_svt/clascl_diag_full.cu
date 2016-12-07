#include "deviceSVD.cuh"
#include "cudacomplex.h"
#include <math.h>
#include <stdio.h>
#include <cuda_runtime_api.h>

#define NB 32

__global__ void
device_batch_clascl_diag_full(int m, int n, int batch_size, const float* D, singlecomplex* A, int lda)
{
    int ind = blockIdx.x * NB + threadIdx.x;

    A += ind;

    if (ind < m) {
        for (int j=0; j < n * batch_size; j++ ) {
            float d = D[j];
            d =  copysignf(sqrt(abs(d)), d);
            A[lda * j] *= d;
        }
    }
}

void batch_clascl_diag_full(
    int m, int n, int batch_size,
    const float *dD,
    singlecomplex *dA, int ldda)
{
    dim3 threads( NB );
    dim3 grid( m / NB + 1 );
    
    device_batch_clascl_diag_full<<< grid, threads, 0>>>(m, n, batch_size, dD, dA, ldda);
}

__global__ void
device_batch_clascl_diag_thres_full(int m, int n, int batch_size, const float* D, singlecomplex* A, int lda, float thres)
{
    int ind = blockIdx.x * NB + threadIdx.x;

    A += ind;

    if (ind < m) {
        for (int j=0; j < n * batch_size; j++ ) {
            float d_squared = D[j];
            float d =  copysignf(sqrt(abs(d_squared)), d_squared);
            float d_thres;
            if(d > thres) {
                d_thres = d - thres;
                float d_quotient = d_thres / d;
                A[lda * j] *= d_quotient;            
            } else if (d < - thres) {
                d_thres = d + thres;
                float d_quotient = d_thres / d;
                A[lda * j] *= d_quotient;
            } else {
                A[lda * j] = 0;
            }
        }
    }
}

void batch_clascl_diag_thres_full(
    int m, int n, int batch_size,
    const float *dD,
    singlecomplex *dA, int ldda,
    float thres)
{

    dim3 threads( NB );
    dim3 grid( m / NB + 1 );

    device_batch_clascl_diag_thres_full<<< grid, threads, 0>>>(m, n, batch_size, dD, dA, ldda, thres);
}