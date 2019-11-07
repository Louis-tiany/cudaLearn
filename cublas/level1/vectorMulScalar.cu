#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define N 6

//compute alpha*x+y,x and y are vector,alpha is a scalar 
int main(void)
{
	cudaError_t cudaStatus;
	cublasStatus_t status;
	cublasHandle_t handle;

	
	float *x;
	float *y;
	x = (float *)malloc(N * sizeof(*x));
	for (int i = 0; i < N; ++i)
		x[i] = (float)i;

	y = (float *)malloc(N * sizeof(*x));
	for (int i = 0; i < N; ++i)
		y[i] = (float)i;

	printf("vector x is:\n");
	for (int i = 0; i < N; ++i)
		printf("%2.0f", x[i]);
	printf("\n");

	printf("vector y is:\n");
	for (int i = 0; i < N; ++i)
		printf("%2.0f", y[i]);
	printf("\n");

	//device code
	float *dx;
	float *dy;
	//alloc memory in GPU for x and y
	cudaStatus = cudaMalloc((void **)&dx, N * sizeof(*x));
	cudaStatus = cudaMalloc((void **)&dy, N * sizeof(*y));

	status = cublasCreate(&handle);
	status = cublasSetVector(N, sizeof(*x), x, 1, dx, 1);
	status = cublasSetVector(N, sizeof(*y), x, 1, dy, 1);
	float alpha = 2.0;

	//compute 
	status = cublasSaxpy(handle, N, &alpha, dx, 1, dy, 1);

	status = cublasGetVector(N, sizeof(float), dy, 1, y, 1);
	printf("compute result\n");

	for (int i = 0; i < N; ++i)
		printf("%2.0f", y[i]);
	printf("\n");

	//free device memory
	cudaFree(dx);
	cudaFree(dy);
	//destroy context
	cublasDestroy(handle);
	//free host memory
	free(x);
	free(y);



	return 0;
}
