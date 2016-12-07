#include "deviceSVD.cuh"
#include "cudacomplex.h"

#define eta 1.9209E-07F			/* eta = the relative machine precision */
#define tol 1.5E-31F			/* tol = the smallest normalized positive number, divided by eta */


//TODO: currently this is row-major, but since we are applying on square matrix it doesn't really matter
__device__ void CSVD(singlecomplex a[][maxA * BATCH_SIZE], int m, int n,
		float s[], singlecomplex u[][maxA * BATCH_SIZE], singlecomplex v[][maxA * BATCH_SIZE], float *work)
/* Singular Value Decomposition, a = u * s * Conj(Tran(v)), a is destroyed by CSVD
   the diagonal matrix s is output as a vector, m must be >= n
   if smaller, a should be filled with zero rows
   this code is adapted from Collected Algorithms from ACM, Algorithm 358
   The transformation Conj(Tran(u)) is applied to the p vectors given in columns
   n, n+1, ..., n+p-1 of matrix a
   See: http://www.scs.fsu.edu/~burkardt/f77_src/toms358/toms358.f
   and: http://www.scs.fsu.edu/~burkardt/f77_src/toms358/toms358_prb.f
*/
{
	// if (threadIdx.x == 0) {
	// 	for (int i=0; i < maxA; i++) {
	// 		for (int j=0; j < maxA; j++) {
	// 			printf("%f,%f ", crealf(access_m(a,i,j)), cimagf(access_m(a,i,j)));
	// 		}
	// 		printf("\n");
	// 	}
	// }
	float *b = work;
	float *c = b + maxA * BATCH_SIZE;
	float *t = c + maxA * BATCH_SIZE;

	float cs, eps, f, g, h;
	int i, j, k, k1, L, L1, nM1;
	singlecomplex q;
	singlecomplex r;
	float sn;
	float w, x, y, z;
	nM1 = n - 1;
	L = 0;
	/*
		HOUSEHOLDER REDUCTION
	*/
	access_a(c,0) = 0.0F;
	k = 0;
	while (1)
	{
		k1 = k + 1;
		/*
			ELIMINATION OF access_m(a,i,k), i = k, ..., m-1
		*/
		z = 0.0F;
		for (i = k; i < m; i++)
			z += norm(access_m(a,i,k));
		access_a(b,k) = 0.0F;
		if (z > tol)
		{
			z = sqrt(z);
			access_a(b,k) = z;
			w = cabsf(access_m(a,k,k));
			q = cOne;
			if (w != 0.0F) { 
				q = access_m(a,k,k) / w;
			}
			access_m(a,k,k) = q * (z + w);
			if (k != nM1)
			{
				for (j = k1; j < n; j++)
				{
					q = cZero;
					for (i = k; i < m; i++)
						q = q + conjf(access_m(a,i,k)) * access_m(a,i,j);
					q /= z * (z + w);
					for (i = k; i < m; i++)
						access_m(a,i,j) -= q * access_m(a,i,k);
				}
			}
			/*
				PHASE TRANSFORMATION
			*/
			q = -conjf(access_m(a,k,k)) / cabsf(access_m(a,k,k));
			for (j = k1; j < n; j++)
				access_m(a,k,j) *= q;
		}
		/*
			ELIMINATION OF access_m(a,k,j), j = k+2, ..., n-1
		*/
		if (k == nM1) break;
		z = 0.0F;
		for (j = k1; j < n; j++)
			z += norm(access_m(a,k,j));
		access_a(c,k1) = 0.0F;
		if (z > tol)
		{
			z = sqrt(z);
			access_a(c,k1) = z;
			w = cabsf(access_m(a,k,k1));
			q = cOne;
			if (w != 0.0F) q = access_m(a,k,k1) / w;
			access_m(a,k,k1) = q * (z + w);
			for (i = k1; i < m; i++)
			{
				q = cZero;
				for (j = k1; j < n; j++)
					q = q + conjf(access_m(a,k,j)) * access_m(a,i,j);
				q /= z * (z + w);
				for (j = k1; j < n; j++)
					access_m(a,i,j) -= q * access_m(a,k,j);
			}
			/*
				PHASE TRANSFORMATION
			*/
			q = -conjf(access_m(a,k,k1)) / cabsf(access_m(a,k,k1));
			for (i = k1; i < m; i++)
				access_m(a,i,k1) *= q;
		}
		k = k1;
	}
	/*
		TOLERANCE FOR NEGLIGIBLE ELEMENTS
	*/
	eps = 0.0F;
	for (k = 0; k < n; k++)
	{
		float b_k = access_a(b,k);
		float c_k = access_a(c,k);
		access_a(s,k) = b_k;
		access_a(t,k) = c_k;
		eps = fmaxf(b_k + c_k, eps);
	}
	eps *= eta;
	/*
		INITIALIZATION OF u AND v
	*/
	for (j = 0; j < m; j++)
	{
		for (i = 0; i < m; i++)
			access_m(u,i,j) = cZero;
		access_m(u,j,j) = cOne;
	}
	// for (j = 0; j < n; j++)
	// {
	// 	for (i = 0; i < n; i++)
	// 		access_m(v,i,j) = cZero;
	// 	access_m(v,j,j) = cOne;
	// }
	/*
		QR DIAGONALIZATION
	*/
	for (k = nM1; k >= 0; k--)
	{
		/*
			TEST FOR SPLIT
		*/
		while (1)
		{
			for (L = k; L >= 0; L--)
			{
				if (fabsf(access_a(t,L)) <= eps) goto Test;
				if (fabsf(access_a(s,L - 1)) <= eps) break;
			}
			/*
				CANCELLATION OF E(L)
			*/
			cs = 0.0F;
			sn = 1.0F;
			L1 = L - 1;
			for (i = L; i <= k; i++)
			{
				f = sn * access_a(t,i);
				access_a(t,i) *= cs;
				if (fabsf(f) <= eps) goto Test;
				h = access_a(s,i);
				w = sqrt(f * f + h * h);
				access_a(s,i) = w;
				cs = h / w;
				sn = -f / w;
				for (j = 0; j < n; j++)
				{
					x = crealf(access_m(u,j,L1));
					y = crealf(access_m(u,j,i));
					access_m(u,j,L1) = x * cs + y * sn;
					access_m(u,j,i) = y * cs - x * sn;
				}
			}
			/*
				TEST FOR CONVERGENCE
			*/
	Test:	w = access_a(s,k);
			if (L == k) break;
			/*
				ORIGIN SHIFT
			*/
			x = access_a(s,L);
			y = access_a(s,k - 1);
			g = access_a(t,k - 1);
			h = access_a(t,k);
			f = ((y - w) * (y + w) + (g - h) * (g + h)) / (2.0F * h * y);
			g = sqrt(f * f + 1.0F);
			if (f < 0.0F) g = -g;
			f = ((x - w) * (x + w) + (y / (f + g) - h) * h) / x;
			/*
				QR STEP
			*/
			cs = 1.0F;
			sn = 1.0F;
			L1 = L + 1;
			for (i = L1; i <= k; i++)
			{
				g = access_a(t,i);
				y = access_a(s,i);
				h = sn * g;
				g = cs * g;
				w = sqrt(h * h + f * f);
				access_a(t,i - 1) = w;
				cs = f / w;
				sn = h / w;
				f = x * cs + g * sn;
				g = g * cs - x * sn;
				h = y * sn;
				y = y * cs;
				// for (j = 0; j < n; j++)
				// {
				// 	x = crealf(access_m(v,j,i - 1));
				// 	w = crealf(access_m(v,j,i));
				// 	access_m(v,j,i - 1) = x * cs + w * sn;
				// 	access_m(v,j,i) = w * cs - x * sn;
				// }
				w = sqrt(h * h + f * f);
				access_a(s,i - 1) = w;
				cs = f / w;
				sn = h / w;
				f = cs * g + sn * y;
				x = cs * y - sn * g;
				for (j = 0; j < n; j++)
				{
					y = crealf(access_m(u,j,i - 1));
					w = crealf(access_m(u,j,i));
					access_m(u,j,i - 1) = y * cs + w * sn;
					access_m(u,j,i) = w * cs - y * sn;
				}
			}
			access_a(t,L) = 0.0F;
			access_a(t,k) = f;
			access_a(s,k) = x;
		}
		/*
			CONVERGENCE
		*/
		// if (w >= 0.0F) continue;
		// access_a(s,k) = -w;
		// for (j = 0; j < n; j++)
		// 	access_m(v,j,k) = -access_m(v,j,k);
	}
	/*
		BACK TRANSFORMATION
	*/
	for (k = nM1; k >= 0; k--)
	{
		// if (access_a(b,k) == 0.0F) continue;
		q = -access_m(a,k,k) / cabsf(access_m(a,k,k));
		for (j = 0; j < m; j++)
			access_m(u,k,j) *= q;
		for (j = 0; j < m; j++)
		{
			q = cZero;
			for (i = k; i < m; i++)
				q = q + conjf(access_m(a,i,k)) * access_m(u,i,j);
			q /= cabsf(access_m(a,k,k)) * access_a(b,k);
			for (i = k; i < m; i++)
				access_m(u,i,j) -= q * access_m(a,i,k);
		}
	}
	// for (k = n - 2; k >= 0; k--)
	// {
	// 	k1 = k + 1;
	// 	// if (access_a(c,k1) == 0.0F) continue;
	// 	q = -conjf(access_m(a,k,k1)) / cabsf(access_m(a,k,k1));
	// 	for (j = 0; j < n; j++)
	// 		access_m(v,k1,j) *= q;
	// 	for (j = 0; j < n; j++)
	// 	{
	// 		q = cZero;
	// 		for (i = k1; i < n; i++)
	// 			q = q + access_m(a,k,i) * access_m(v,i,j);
	// 		q /= (cabsf(access_m(a,k,k1)) * access_a(c,k1));
	// 		for (i = k1; i < n; i++)
	// 			access_m(v,i,j) -= q * conjf(access_m(a,k,i));
	// 	}
	// }

} /* CSVD */


__global__ void batchSVD(singlecomplex *a, int m, int n, float *s, 
	singlecomplex *u, singlecomplex *v, float *work) {
	a += blockIdx.x * blockDim.x + threadIdx.x;
	u += blockIdx.x * blockDim.x + threadIdx.x;
	v += blockIdx.x * blockDim.x + threadIdx.x;
	s += blockIdx.x * blockDim.x + threadIdx.x;
	work += blockIdx.x * blockDim.x + threadIdx.x;
  	CSVD((singlecomplex(*)[maxA * BATCH_SIZE])a, m, n, s, (singlecomplex(*)[maxA * BATCH_SIZE])u, (singlecomplex(*)[maxA * BATCH_SIZE])v, work);
}

