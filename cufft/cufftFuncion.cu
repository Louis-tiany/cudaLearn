#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include"device_launch_parameters.h"
#include "cublas_v2.h"
#include "cufft.h"
#include <iostream>
#include <cuda.h>

void checkError(cudaError_t cudaStatus)
{
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "line number:" << __LINE__ << "\n";
		std::cout << "error:" << cudaGetErrorString(cudaStatus) << "\n";
	}
}

void fftStatus(cufftResult cufftStatus)
{
	if (cufftStatus != CUFFT_SUCCESS)
	{
		std::cout << "line number:" << __LINE__ << "\n";
	}
}



void fft1D(cuComplex *data,cuComplex *result,int num)
{
	int Length = num;
	cuComplex *dData;
	cuComplex *dResult;
	cudaMalloc((void **)&dData, Length * sizeof(float));
	cudaMalloc((void **)&dResult, Length * sizeof(cuComplex));
	//create handle of fft real--->complex
	cufftHandle fftHandle;
	fftStatus(cufftPlan1d(&fftHandle, Length, CUFFT_C2C, 1));

	//send data to device
	cudaMemcpy(dData, data, Length * sizeof(float), cudaMemcpyHostToDevice);

	cufftExecC2C(fftHandle, dData, dResult,CUFFT_FORWARD);
	//copy memory from device to host
	checkError(cudaMemcpy(result, dResult, Length*sizeof(cufftComplex), cudaMemcpyDeviceToHost));

	cufftDestroy(fftHandle);
	cudaFree(dData);
	cudaFree(dResult);
}

//use for ifft function
__global__ void fftNorm(cuComplex *data,int length)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	data[idx].x /= length;
	data[idx].y /= length;
}


void fftInverse1D(cuComplex *data,cuComplex *result,int num)
{
	int Length = num;
	cuComplex * dData;
	cuComplex *dResult;
	cufftHandle fftInverseHandle;
	fftStatus(cufftPlan1d(&fftInverseHandle, Length, CUFFT_C2C, 1));

	cudaMalloc((void **)&dData, Length * sizeof(cuComplex));
	cudaMalloc((void **)&dResult, Length  * sizeof(cuComplex));

	//send data
	checkError(cudaMemcpy(dData, data, Length * sizeof(cuComplex), cudaMemcpyHostToDevice));
	//compute
	fftStatus(cufftExecC2C(fftInverseHandle, dData, dResult,CUFFT_INVERSE));
	//normlizing
	dim3 grid(8);
	dim3 block(Length / 8);
	fftNorm << <grid, block >> > (dResult, Length);
	//fetch data
	checkError(cudaMemcpy(result, dResult, Length * sizeof(float), cudaMemcpyDeviceToHost));

	cufftDestroy(fftInverseHandle);
	cudaFree(dData);
	cudaFree(dResult);
}



//get length of complex array's abs
//input *src and *dest are device memory's pointer
__global__ void complexAbs(cuComplex *src, float *dest)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	dest[idx] = cuCabsf(src[idx]);
}

//src and dest are host memory's pointer
void complexNorm(cuComplex *src, float *dest, int num)
{
	int Length = num;
	//the size of gird and block should be changed depending input data
	dim3 grid(8);
	dim3 block(Length / 8);

	cufftComplex *dSrc;
	float *dDest;
	cudaMalloc((void **)&dSrc, Length * sizeof(cufftComplex));
	cudaMalloc((void **)&dDest, Length * sizeof(float));
	//send data
	checkError(cudaMemcpy(dSrc, src, Length * sizeof(cufftComplex), cudaMemcpyHostToDevice));
	//compute
	complexAbs << <grid, block >> > (dSrc, dDest);
	//fetch data
	checkError(cudaMemcpy(dest, dDest, Length * sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(dSrc);
	cudaFree(dDest);
}

__global__ void complexConjUtil(cuComplex *src, cuComplex *dest)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	dest[idx].x = src[idx].x;
	dest[idx].y = -src[idx].y;
}

//src and dest is host's memory pointer
void complexConj(cuComplex *src, cuComplex *dest,int num)
{

	cuComplex *dSrc, *dDest;

	cudaMalloc((void **)&dSrc, num * sizeof(cuComplex));
	cudaMalloc((void **)&dDest, num * sizeof(cuComplex));

	int Length = num;
	dim3 grid(8);
	dim3 block(Length / 8);
	//send data
	checkError(cudaMemcpy(dSrc, src, Length * sizeof(cufftComplex), cudaMemcpyHostToDevice));
	//compute
	complexConjUtil << <grid, block >> > (dSrc, dDest);
	//fetch data
	checkError(cudaMemcpy(dest, dDest, Length * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
	cudaFree(dSrc);
	cudaFree(dDest);
}


int main()
{

	const int Length = 16;

	cuComplex *data;
	data = (cuComplex *)malloc(Length * sizeof(cuComplex));
	for (int i = 0; i < Length; ++i)
	{
		data[i].x = (float)i;
		data[i].y = (float)i;
	}
	printf("data:\n");
	for (int i = 0; i < Length; ++i)
		printf("%f+j%f\n", data[i].x, data[i].y);
	printf("\n");


	cuComplex *dataConj;
	dataConj = (cuComplex *)malloc(Length * sizeof(cuComplex));
	complexConj(data, dataConj, Length);
	printf("data's Conjugate:\n");
	for (int i = 0; i < Length; ++i)
		printf("%f+j%f\n", dataConj[i].x, dataConj[i].y);
	printf("\n");
	free(dataConj);



	cuComplex *result;
	result = (cuComplex *)malloc(Length * sizeof(cuComplex));

	fft1D(data, result, Length);

	printf("fft result:\n");
	for (int i = 0; i < Length; ++i)
		printf("%f+j%f\n", result[i].x, result[i].y);
	printf("\n");

	fftInverse1D(result, data, Length);
	printf("inverse fft result:\n");
	for (int i = 0; i < (Length); ++i)
		printf("%f+j%f\n", data[i].x,data[i].y);
	printf("\n");

	free(data);
	free(result);

	printf("\n-------------------------------\n");



	return 0;
}

