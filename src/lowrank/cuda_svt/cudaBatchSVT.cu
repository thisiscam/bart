#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include "cudacomplex.h"
#include "cudaBatchSVT.h"
#include "deviceSVD.cuh" 
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include "num/cuda_commons.h"
#include <iostream>

#ifndef BATCH_SVT_USE_FIXED_SIZE
#include "nvrtc_batchCHQL.cuh"
#endif

// events for timing
static cudaEvent_t startEvent, stopEvent;
static float ms;
static cublasStatus_t ret;
extern cublasHandle_t cublas_handle;

void init_batch_svt(int devId)
{
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );
}

__attribute__((constructor)) static void init() 
{
  init_batch_svt(0); 
}

void cublas_transpose(cublasHandle_t handle, int m, int n, const cuComplex* d_a, cuComplex* d_at)
{
  //Transpose
  const cuComplex alpha = make_cuComplex(1.0f,0.0f);
  const cuComplex beta  = make_cuComplex(0.0f,0.0f);
  //Perform operation with cublas
  checkCuda( cudaEventRecord(startEvent, 0) );
  ret = cublasCgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                    m, n, 
                    &alpha, d_a, n, &beta, NULL, m,
                    d_at, m);

  if (ret != CUBLAS_STATUS_SUCCESS)
  {
      printf("cublasCgeamm returned error code %d, line(%d)\n", ret, __LINE__);
      exit(EXIT_FAILURE);
  }
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  printf("Transpose took %f\n", ms);
}

void cublas_transpose_float(cublasHandle_t handle, int m, int n, const float* d_a, float* d_at)
{
  //Transpose
  const float alpha = 1.0f;
  const float beta  = 0.0f;
  //Perform operation with cublas
  checkCuda( cudaEventRecord(startEvent, 0) );
  ret = cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                    m, n, 
                    &alpha, d_a, n, &beta, NULL, m,
                    d_at, m);

  if (ret != CUBLAS_STATUS_SUCCESS)
  {
      printf("cublasSgeamm returned error code %d, line(%d)\n", ret, __LINE__);
      exit(EXIT_FAILURE);
  }
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  printf("Transpose float took %f\n", ms);
}

void cublas_batch_dgemm(cublasHandle_t handle, 
                        cublasOperation_t transa, cublasOperation_t transb, 
                        int m, int n, int k, 
                        const cuComplex** a, int lda, 
                        const cuComplex** b, int ldb, 
                        cuComplex **c, int ldc, 
                        int batch_size)
{
  //Transpose
  const cuComplex alpha = make_cuComplex(1.0f,0.0f);
  const cuComplex beta  = make_cuComplex(0.0f,0.0f);
  //Perform operation with cublas
  checkCuda( cudaEventRecord(startEvent, 0) );
  ret = cublasCgemmBatched(handle, transa, transb, 
                    m, n, k,
                    &alpha, 
                    a, lda, 
                    b, ldb,
                    &beta, 
                    c, ldc,
                    batch_size);

  if (ret != CUBLAS_STATUS_SUCCESS)
  {
      printf("cublasCgemm returned error code %d, line(%d)\n", ret, __LINE__);
      exit(EXIT_FAILURE);
  }
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  printf("batch multiplication took %f\n", ms);
}

#define CREATE_BATCH_PTR(start, step, batch_size, dst) create_batch_ptr((void*)start, step, batch_size, (void**)dst)
void create_batch_ptr(void* start, size_t step, int batch_size, void** dst)
{
  int batch_ptrs_size = sizeof(void*) * batch_size;
  void** ptrs = (void**)malloc(batch_ptrs_size);
  for(int i=0; i < batch_size; i++)
  {
    ptrs[i] = ((char*)start) + step * i;
  }
  checkCuda( cudaMemcpy(dst, ptrs, batch_ptrs_size, cudaMemcpyHostToDevice) );
}

/*TODO include debug routine somwhere*/
// void WriteMatrixDevice(singlecomplex *matrix, int m, int n)
// {
//   int size = sizeof(singlecomplex) * m * n;
//   singlecomplex *host_copy = (singlecomplex*)malloc(size);
//   checkCuda( cudaMemcpy(host_copy, matrix, size, cudaMemcpyDeviceToHost) );
//   cudaDeviceSynchronize();
//   WriteMatrix ((singlecomplex(*)[maxA])host_copy, m, n);
//   free(host_copy);
// }

// void WriteDiagDevice(float *diag, int m)
// {
//   int size = sizeof(float) * m;
//   float *host_copy = (float*)malloc(size);
//   checkCuda( cudaMemcpy(host_copy, diag, size, cudaMemcpyDeviceToHost) );
//   for(int i=0; i < m; i++) {
//     printf("%f\n", host_copy[i]);
//   }
//   free(host_copy);
// }

void cuda_batch_svt(int m, int n, int batch_size, float tau, _Complex float *_d_tau_a, const _Complex float *_d_a)
{
  const singlecomplex *d_a = (const singlecomplex*)_d_a;
  singlecomplex *d_tau_a = (singlecomplex*)_d_tau_a;

#ifdef BATCH_SVT_USE_FIXED_SIZE
  assert(m == maxA && "Batch SVT is compiled with fixed size, and you passed in something different");
  assert(batch_size == BATCH_SIZE && "Batch SVT is compiled with fixed size, and you passed in something different");
#else

#endif
  bool m_gt_n = m > n;
  if(m_gt_n) {
    int tmp = n;
    n = m;
    m = tmp;
  }
  const long block_a_mem_size = m * n * sizeof(singlecomplex);
  const long block_b_mem_size = m * m * sizeof(singlecomplex);
  const long block_s_mem_size = m * sizeof(singlecomplex);
  const long b_mem_size = m * m * sizeof(singlecomplex) * batch_size;
  const long s_mem_size = m * sizeof(singlecomplex) * batch_size;
  const long batch_ptrs_size = sizeof(const cuComplex*) * batch_size;

  /* Compute B = A * A^T */
  singlecomplex *d_b;
  const cuComplex **d_a_ptr; 
  cuComplex **d_b_ptr;
  checkCuda( cudaMalloc(&d_b, b_mem_size) );
  checkCuda( cudaMalloc(&d_a_ptr, batch_ptrs_size) );
  checkCuda( cudaMalloc(&d_b_ptr, batch_ptrs_size) );
  CREATE_BATCH_PTR(d_a, block_a_mem_size, batch_size, d_a_ptr);
  CREATE_BATCH_PTR(d_b, block_b_mem_size, batch_size, d_b_ptr);
  if(m_gt_n) {
    cublas_batch_dgemm(cublas_handle, CUBLAS_OP_C, CUBLAS_OP_N, m, m, n, d_a_ptr, n, d_a_ptr, n, d_b_ptr, m, batch_size);
  } else {
    cublas_batch_dgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_C, m, m, n, d_a_ptr, m, d_a_ptr, m, d_b_ptr, m, batch_size);
  }
  checkCuda( cudaFree(d_b_ptr) );

  /* Enter interleaved mode */
  singlecomplex *d_bt;
  checkCuda( cudaMalloc(&d_bt, b_mem_size) );
  cublas_transpose(cublas_handle, batch_size, m * m, (const cuComplex*)d_b, (cuComplex*)d_bt);

  /* Compute Core SVD */
  singlecomplex *d_ut = d_b;
  float *d_st, *d_work;
  checkCuda( cudaMalloc(&d_st, s_mem_size) );
  checkCuda( cudaMalloc(&d_work, s_mem_size * 3) );
#ifdef BATCH_SVT_USE_FIXED_SIZE
  batchCHQL<<<batch_size/INTERLEAVE,INTERLEAVE>>>(d_bt, m, d_st, d_ut, d_work);
#else
  nvrtc_batchCHQL(d_bt, m, d_st, d_ut, d_work, batch_size);
#endif 

  /* Exit interleaved mode */
  singlecomplex *d_u = d_bt;
  float *d_s;
  checkCuda( cudaMalloc(&d_s, s_mem_size) );
  cublas_transpose(cublas_handle, m * m, batch_size, (const cuComplex*)d_ut, (cuComplex*)d_u);
  cublas_transpose_float(cublas_handle, m, batch_size, (const float*)d_st, (float*)d_s);

  /* Compute U * tau(S) * U^T. Currently this is not optimal */
  singlecomplex *d_u_scal_s;
  singlecomplex *d_usut = d_ut;
  checkCuda( cudaMalloc(&d_u_scal_s, b_mem_size) );
  checkCuda( cudaMemcpy(d_u_scal_s, d_u, b_mem_size, cudaMemcpyDeviceToDevice) );
  batch_clascl_diag_thres_full(m, m, batch_size, d_s, d_u_scal_s, m, tau);
  checkCuda( cudaFree(d_s) );

  const cuComplex **d_u_scal_s_ptr,  **d_u_ptr;
  cuComplex **dusut_ptr;
  checkCuda( cudaMalloc(&d_u_scal_s_ptr, batch_ptrs_size) );
  checkCuda( cudaMalloc(&d_u_ptr, batch_ptrs_size) );
  checkCuda( cudaMalloc(&dusut_ptr, batch_ptrs_size) );
  CREATE_BATCH_PTR(d_u_scal_s, block_b_mem_size, batch_size, d_u_scal_s_ptr);
  CREATE_BATCH_PTR(d_u, block_b_mem_size, batch_size, d_u_ptr);
  CREATE_BATCH_PTR(d_usut, block_b_mem_size, batch_size, dusut_ptr);
  cublas_batch_dgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_C, m, m, m, d_u_scal_s_ptr, m, d_u_ptr, m, dusut_ptr, m, batch_size);

  /* Compute tau(USU^T) * A */
  cuComplex **d_tau_a_ptr;
  checkCuda( cudaMalloc(&d_tau_a_ptr, batch_ptrs_size) );
  CREATE_BATCH_PTR(d_tau_a, block_a_mem_size, batch_size, d_tau_a_ptr);
  if(m_gt_n) {
    cublas_batch_dgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                      n, m, m, 
                      (const cuComplex**)d_a_ptr, n, 
                      (const cuComplex**)dusut_ptr, m, 
                      d_tau_a_ptr, n, batch_size);
  } else {
    cublas_batch_dgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                      m, n, m, 
                      (const cuComplex**)dusut_ptr, m, 
                      (const cuComplex**)d_a_ptr, m, 
                      d_tau_a_ptr, m, batch_size);
  }

  checkCuda( cudaFree(d_u_ptr) );
  checkCuda( cudaFree(d_usut) );
  checkCuda( cudaFree(d_u_scal_s_ptr) );
  checkCuda( cudaFree(dusut_ptr));
  checkCuda( cudaFree(d_tau_a_ptr));
  checkCuda( cudaFree(d_a_ptr) );
  checkCuda( cudaFree(d_u) );
}
