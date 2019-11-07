#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define N 6


int main(void)
{
	cudaError_t cudaStatus;
	cublasStatus_t status;
	cublasHandle_t handle;

	float *vec;
	vec = (float *)malloc(N * sizeof(float));
	for (int i = 0; i < N; ++i)
		vec[i] = (float)i;
	printf("x:\n");
	for (int i = 0; i < N; ++i)
		printf("%4.0f",vec[i]);
	printf("\n");

	//device codes
	float *dVec;
	cudaStatus = cudaMalloc((void **)&dVec, N * sizeof(float));
	status = cublasCreate(&handle);
	status = cublasSetVector(N, sizeof(*vec), vec, 1, dVec, 1);

	int result;//note max value's index
	status = cublasIsamax(handle, N, dVec, 1, &result);
	
	printf(" m a x | vec [ i ] | : %4 . 0 f \ n",fabs(vec[result-1]));
	
	//free device memory
	cudaFree(dVec);
	//destory cublas context
	cublasDestroy(handle);

	//free host memory 
	free(vec);

	return 0;
}
