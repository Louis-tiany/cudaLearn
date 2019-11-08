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
	x = (float *)malloc(N * sizeof(*x));
	for (int i = 0; i < N; ++i)
		x[i] = float(i);
	
	printf("x:");
	for (int i = 0; i < N; ++i)
		printf("%4.0f", x[i]);
	printf("\n");

	float *y;
	y = (float *)malloc(N * sizeof(*x));


	//device codes

	float *dx, *dy;
	cudaStatus = cudaMalloc((void **)dx, N * sizeof(*x));
	cudaStatus = cudaMalloc((void **)dy, N * sizeof(*y));

	status = cublasCreate(&handle);//create context
	status = cublasSetVector(N, sizeof(*x), x, 1, dx, 1);//x->dx

	//copy values:dx--->dy
	status = cublasScopy(handle, N, dx, 1, dy, 1);

	//copy device values to host values: dy --->y
	status = cublasSetVector(N, sizeof(float), dy, 1, y, 1);

	printf("after copy values\n");
	for (int i = 0; i < N; ++i)
		printf("%4.0f", y[i]);
	printf("\n");


	//free device memory
	cudaFree(dx);
	cudaFree(dy);
	
	//free cuda context
	cublasDestroy(handle);

	//free host memory
	free(x);
	free(y);


	return 0;
}

