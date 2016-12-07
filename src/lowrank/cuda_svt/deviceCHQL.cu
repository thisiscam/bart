#include "deviceSVD.cuh"

#ifndef __CUDACC_RTC__
#include <stdio.h>
#endif

#define n maxA

__device__ void htridi ( int np, singlecomplex a[][maxA * BATCH_SIZE], float d[maxA * BATCH_SIZE], 
  float e[maxA], float tau[2][maxA * BATCH_SIZE])
{
  float f; float fi;
  float g; float gi;
  float h; float hh;
  float si;
  int i;
  int j;
  int k;
  int l;
  float scale;

  access_m(tau,0,n-1) = 1;
  access_m(tau,1,n-1) = 0;

  for (i=0; i < n; i++) {
    access_a(d,i) = crealf(access_m(a,i,i));
  }

  for ( i = n - 1; i >= 0; i-- )
  {
    l = i - 1;
    h = 0.0;
    /*Scale row.*/
    scale = 0.0;
    if (l >= 0) {
      for ( k = 0; k <= l; k++ )
      {
        scale = scale + fabsf(crealf(access_m(a,i,k))) + fabsf(cimagf(access_m(a,i,k)));
      }

      if ( fabsf(scale) < 1e-8 )
      {
        access_m(tau,0,l) = 1;
        access_m(tau,1,l) = 0;
        e[i] = 0;
      } else {
        for ( k = 0; k <= l; k++ )
        {
          access_m(a,i,k) /= scale;
          h = h + norm(access_m(a,i,k));
        }
        g = sqrt(h);
        e[i] = scale * g;
        f = cabsf(access_m(a,i,l));
        /*For next diagonal element of matrix*/
        if (f != 0) {
          access_m(tau,0,l) = (cimagf(access_m(a,i,l)) * access_m(tau,1,i) - crealf(access_m(a,i,l)) * access_m(tau,0,i)) / f;
          si = (crealf(access_m(a,i,l)) * access_m(tau,1,i) + cimagf(access_m(a,i,l)) * access_m(tau,0,i)) / f;
          h = h + f * g;
          g = 1 + g / f;
          access_m(a,i,l) *= g;
          if (l == 0) { goto scale_a;}
        } else {
          access_m(tau,0,l) = - access_m(tau,0,i);
          si = access_m(tau,1,i);
          access_m(a,i,l).re() = g; //set real part of access_m(a,i,l)
        }
        f = 0.0;
        for (j = 0; j <= l; j++) {
          g = 0;
          gi = 0;
          /*Form element of a*u*/
          for (k = 0; k <= j; k++) {
            g = g + crealf(access_m(a,j,k)) * crealf(access_m(a,i,k)) + cimagf(access_m(a,j,k)) * cimagf(access_m(a,i,k));
            gi = gi - crealf(access_m(a,j,k)) * cimagf(access_m(a,i,k)) + cimagf(access_m(a,j,k)) * crealf(access_m(a,i,k));
          }
          for (k = j + 1; k <= l; k++) {
            g = g + crealf(access_m(a,k,j)) * crealf(access_m(a,i,k)) - cimagf(access_m(a,k,j)) * cimagf(access_m(a,i,k));
            gi = gi - crealf(access_m(a,k,j)) * cimagf(access_m(a,i,k)) - cimagf(access_m(a,k,j)) * crealf(access_m(a,i,k));
          }
          /*Form element of p*/
          e[j] = g / h;
          access_m(tau,1,j) = gi / h;
          f = f + e[j] * crealf(access_m(a,i,j)) - access_m(tau,1,j) * cimagf(access_m(a,i,j));
        }
        hh = f / (h + h);
        /*Form reduced a*/
        for (j = 0; j <= l; j++) {
          f = crealf(access_m(a,i,j));
          g = e[j] - hh * f;
          e[j] = g;
          fi = - cimagf(access_m(a,i,j));
          gi = access_m(tau,1,j) - hh * fi;
          access_m(tau,1,j) = -gi;
          for (k = 0; k <= j; k++) {
            access_m(a,j,k) += singlecomplex( - f * e[k] - g * crealf(access_m(a,i,k)) + fi * access_m(tau,1,k) + gi * cimagf(access_m(a,i,k)),
             - f * access_m(tau,1,k) - g * cimagf(access_m(a,i,k)) - fi * e[k] - gi * crealf(access_m(a,i,k)));
          }
        }
scale_a:
        for (k = 0; k <= l; k++) {
          access_m(a,i,k) *= scale;
        }
        access_m(tau,1,l) = -si;
      }
    } else {
      e[i] = 0;
    }
    hh = access_a(d,i);
    access_a(d,i) = crealf(access_m(a,i,i));
    access_m(a,i,i) = singlecomplex(hh, scale * sqrt(h));
  }
  return;
}
/******************************************************************************/


__device__ int tql2 ( int np, float d[maxA * BATCH_SIZE], float e[maxA], singlecomplex z[][maxA * BATCH_SIZE] )
{
  float c;
  float c2;
  float c3;
  float dl1;
  float el1;
  float f;
  float g;
  float h;
  int i;
  int ierr;
  int ii;
  int j;
  int k;
  int l;
  int l1;
  int l2;
  int m;
  int mml;
  float p;
  float r;
  float s;
  float s2;
  float t;
  float tst1;
  float tst2;

  ierr = 0;

  if ( n == 1 )
  {
    return ierr;
  }

  for ( i = 1; i < n; i++ )
  {
    e[i-1] = e[i];
  }

  f = 0.0;
  tst1 = 0.0;
  e[n-1] = 0.0;

  for ( l = 0; l < n; l++ )
  {
    j = 0;
    h = fabsf ( access_a(d,l) ) + fabsf ( e[l] );
    tst1 = fmaxf ( tst1, h );
/*
  Look for a small sub-diagonal element.
*/
    for ( m = l; m < n; m++ )
    {
      tst2 = tst1 + fabsf ( e[m] );
      if ( tst2 == tst1 )
      {
        break;
      }
    }
    // printf("m %d\n", m);
    //Since GPU does not divergent code, we always use m = n - 1 and hope it won't hurt accuracy too much
    // m = n - 1;
    if ( m != l )
    {
      for ( ; ; )
      {
        if ( 30 <= j )
        {
          ierr = l + 1;
          return ierr;
        }

        j = j + 1;
/*
  Form shift.
*/
        l1 = l + 1;
        l2 = l1 + 1;
        g = access_a(d,l);
        p = ( access_a(d,l1) - g ) / ( 2.0 * e[l] );
        r = sqrt ( p * p + 1.0 );
        access_a(d,l) = e[l] / ( p + copysignf(r, p) );
        access_a(d,l1) = e[l] * ( p + copysignf(r, p) );
        dl1 = access_a(d,l1);
        h = g - access_a(d,l);
        for ( i = l2; i < n; i++ )
        {
          access_a(d,i) = access_a(d,i) - h;
        }
        f = f + h;
/*
  QL transformation.
*/
        p = access_a(d,m);
        c = 1.0;
        c2 = c;
        el1 = e[l1];
        s = 0.0;
        mml = m - l;

        for ( ii = 1; ii <= mml; ii++ )
        {
          c3 = c2;
          c2 = c;
          s2 = s;
          i = m - ii;
          g = c * e[i];
          h = c * p;
          r = sqrt ( p * p  + e[i] * e[i] );
          e[i+1] = s * r;
          s = e[i] / r;
          c = p / r;
          p = c * access_a(d,i) - s * g;
          access_a(d,i+1) = h + s * ( c * g + s * access_a(d,i) );
/*
  Form vector.
*/
          for ( k = 0; k < n; k++ )
          {
            h = crealf(access_m(z,i+1,k));
            access_m(z,i+1,k).re() = s * crealf(access_m(z,i,k)) + c * h;
            access_m(z,i,k).re() = c * crealf(access_m(z,i,k)) - s * h;
          }
        }
        p = - s * s2 * c3 * el1 * e[l] / dl1;
        e[l] = s * p;
        access_a(d,l) = c * p;
        tst2 = tst1 + fabsf ( e[l] );

        if ( tst2 <= tst1 )
        {
          break;
        }
      }
    }
    access_a(d,l) = access_a(d,l) + f;
  }
/*
  Order eigenvalues and eigenvectors.
*/
  // for ( ii = 1; ii < n; ii++ )
  // {
  //   i = ii - 1;
  //   k = i;
  //   p = access_a(d,i);
  //   for ( j = ii; j < n; j++ )
  //   {
  //     if ( access_a(d,j) < p )
  //     {
  //       k = j;
  //       p = access_a(d,j);
  //     }
  //   }

  //   if ( k != i )
  //   {
  //     access_a(d,k) = access_a(d,i);
  //     access_a(d,i) = p;
  //     for ( j = 0; j < n; j++ )
  //     {
  //       t        = access_m(z,j,i);
  //       access_m(z,i,j) = access_m(z,k,j);
  //       access_m(z,k,j) = t;
  //     }
  //   }
  // }
  return ierr;
}


__device__ void htribk(int np, const singlecomplex* __restrict__ a, const float* __restrict__ tau, int m, singlecomplex z[][maxA * BATCH_SIZE] ) {
  for (int k = 0; k < n; k++) {
    for (int j = 0; j < m; j++) {
      access_m(z,k,j) = singlecomplex(crealf(access_m(z,k,j)) * res_acc_m(tau,0,j), -crealf(access_m(z,k,j)) * res_acc_m(tau,1,j));
    }
  }

  for (int i=1; i < n; i++) {
    int l = i - 1;
    float h = cimagf(res_acc_m(a,i,i));
    if (fabsf(h) > 1e-10) {
      for (int j = 0; j < m; j++) {
        singlecomplex s = cZero;
        for (int k=0; k <= l; k++) {
          s += res_acc_m(a,i,k) * access_m(z,j,k);
        }
        s /= (h * h);
        for (int k=0; k <= l; k++) {
          access_m(z,j,k) -= s * conjf(res_acc_m(a,i,k));
        }
      }
    }
  }
}

__device__ void CHQL(singlecomplex a[][maxA * BATCH_SIZE], int np,
float s[], singlecomplex u[][maxA * BATCH_SIZE], float *shared_work, float *work) {
  float *e = shared_work;
  float *tau = work;
  htridi (n, a, s, e, (float(*)[maxA*BATCH_SIZE])tau);

  /* Let U = eye. 
      note that for some complicated reason, 
      combining the two loops will cause the program to fail. 
  */
  for (int i=0; i<n; i++) {
    for(int j=0; j<n; j++) {      
      access_m(u,i,j) = cZero;
    }
  }
  for (int i=0; i<n; i++) {
    access_m(u,i,i) = cOne;
  }

  tql2 (n, s, e, u);
  htribk (n, (singlecomplex*)a, tau, n, u);
}

__global__ void __launch_bounds__(INTERLEAVE) batchCHQL(singlecomplex *a, int np, float *s, 
  singlecomplex *u, float *work) {

#ifdef __CUDACC_RTC__
  /* 
    If dynamic compile, we allow batch_size not to be multiple of INTERLEAVE,
    threads with id \in [round_up(batch_size/INTERLEAVE), batch_size) should not do work
  */
  if (blockIdx.x * blockDim.x + threadIdx.x >= BATCH_SIZE) {
    return;
  }
#endif

  /* Prepare work array pointers for each thread */
  a += blockIdx.x * blockDim.x + threadIdx.x;
  u += blockIdx.x * blockDim.x + threadIdx.x;
  s += blockIdx.x * blockDim.x + threadIdx.x;
  work += blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float shared_work[maxA * INTERLEAVE];
  float *private_shared_work = shared_work + maxA * threadIdx.x;

  CHQL((singlecomplex(*)[maxA * BATCH_SIZE])a, n, s, (singlecomplex(*)[maxA * BATCH_SIZE])u, private_shared_work, work);
}

