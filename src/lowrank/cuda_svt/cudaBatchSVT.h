#include <complex.h>
/* 
*	Perform Singular Value (Soft)Thresolding on batch of complex single matrices 
*
*	@Params
*	[In, device_ptr] d_a : concatentated array of matrices SVT should be performed on. 
			d_a should be of shape (m, n, batch_size).
*	[In] m, n: block matrix shape
*	[Out, device_ptr] d_tau_a: output tau(A) matrices. d_tau_a on return will be same shape as d_a
*	[In] tau: soft threshold tau 
*	[In] batch_size: number of batch of matrices to compute SVT on
*	
*	Notes: 
*	- if during compilation BATCH_SVT_USE_FIXED_SIZE is set, the routine will only accept 
*	a fixed m and batch_size, as defined by macros maxA and BATCH_SIZE; attempts to call 
*	this routine with a different m or batch_size will error.
*
*	- If you want dynamic behavior, do not set BATCH_SVT_USE_FIXED_SIZE and the routine will 
*	dynamically(aka Just-In-Time) compile specialized cuda kernels according to the arguments.
*	Currently, the JITted code is cached using a simple LRU cache.
*
**/
#ifdef __cplusplus
extern "C" {
#endif
void cuda_batch_svt( 
					int m, int n,
					int batch_size,
					float tau,
					_Complex float *d_tau_a,
					const _Complex float *d_a
					);
#ifdef __cplusplus
}
#endif

