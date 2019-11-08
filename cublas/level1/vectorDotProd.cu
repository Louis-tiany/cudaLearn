#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define N 6

//compute x*y,x and y are vector,result is dot product of x and y and euclidean norm of x and y
int main(void)
{
	cudaError_t cudaStatus;
	cublasStatus_t status;
	cublasHandle_t handle;

	float *x, *y;
	x = (float *)malloc(N * sizeof(*x));
	y = (float *)malloc(N * sizeof(*y));

	for (int i = 0; i < N; ++i)
	{
		x[i] = float(i);
		y[i] = float(i);
	}

	printf("vec x and y is:\n");
	for (int i = 0; i < N; i++)
	{
		printf("%4.0f%4.0f", x[i], y[i]);
		printf("\n");
	}


	//device codes
	float *dx, *dy;
	cudaStatus=cudaMalloc((void **)&dx, N * sizeof(float));
	cudaStatus = cudaMalloc((void **)&dy, N * sizeof(float));
	status=cublasCreate(&handle);
	status = cublasSetVector(N, sizeof(*x), x, 1, dx, 1);
	status = cublasSetVector(N, sizeof(*x), x, 1, dx, 1);

	

	//get dot product
	float result;
	cublasSdot(handle, sizeof(float), dx, 1, dy, 1,&result);

	//get euclidean norm of vector dx and dy
	float normX;
	cublasSnrm2(handle,sizeof(float),dx,1,&normX);

	float normY;
	cublasSnrm2(handle, sizeof(float), dy, 1, &normY);


	printf("dot product of x and y is :\t");
	printf("&6.0f\n",result);

	printf("euclidean norm of x is : \t");
	printf("&6.0f\n", normX);

	printf("euclidean norm of y is : \t");
	printf("&6.0f\n", normY);

	//free device memory

	cudaFree(dx);
	cudaFree(dy);
	cublasDestroy(handle);

	//free host memory
	free(x);
	free(y);


	return 0;
}
