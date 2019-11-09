#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"


//result=alpha*A*x+beta*y,A is a matrix ,x and y is vector
void maxtrixMulVec(int row, int col, float al,float be,float A[], float vecX[], float vecY[], float res[])
{

	cudaError_t cudaStatus;
	cublasStatus_t status;
	cublasHandle_t handle;

	float alpha = al;
	float beta = be;

	//device code
	float *dA;
	float *dX;
	float *dY;

	cudaStatus = cudaMalloc((void **)&dA, row*col * sizeof(float));
	cudaStatus = cudaMalloc((void **)&dX, col * sizeof(float));
	cudaStatus = cudaMalloc((void **)&dY, row * sizeof(float));

	status = cublasCreate(&handle);
	status = cublasSetMatrix(row, col, sizeof(float), A, row, dA, row);
	status = cublasSetVector(col, sizeof(float), vecX, 1, dX, 1);
	status = cublasSetVector(row, sizeof(float), vecY, 1, dY, 1);

	//compute
	status = cublasSgemv(handle, CUBLAS_OP_N, row, col, &alpha, dA, row, dX, 1, &beta, dY, 1);

	status = cublasGetVector(row, sizeof(float), dY, 1, vecY, 1);

	printf("y after sgemv:\n");
	for (int i = 0; i < row; i++)
		printf("%f\t", vecY[i]);

	cudaFree(dA);
	cudaFree(dX);
	cudaFree(dY);
	cublasDestroy(handle);
}



int main()
{
	//matrix * vector test code
	int row=6;
	int col=5;

	float *a;
	float *x2;
	float *y2;
	a = (float *)malloc(row*col * sizeof(float));
	x2 = (float *)malloc(col * sizeof(float));
	y2 = (float *)malloc(row * sizeof(float));
	
	int ind = 11;

	for (int j = 0; j < col; ++j)
		for (int i = 0; i < row; ++i)
			a[IDX2C(i, j, row)] = (float)ind++;
	printf("matrix :\n");
	for (int i = 0; i < row; ++i)
	{
		for (int j = 0; j < col; ++j)
			printf("%f\t", a[IDX2C(i,j,row)]);
		printf("\n");
	}

	for (int i = 0; i < col; ++i)
		x2[i] = (float)1.0f;
	printf("vec x is:\n");
	for (int i = 0; i < col; ++i)
		printf("%f\t", x2[i]);

	for (int i = 0; i < row; ++i)
		y2[i] = 2.0f;
	printf("vec y is:\n");
	for (int i = 0; i < row; ++i)
		printf("%f\t", y2[i]);
	
	maxtrixMulVec(row, col, 1, 1, a, x2, y2, y2);


	free(a);
	free(x2);
	free(y2);




	return 0;
}

