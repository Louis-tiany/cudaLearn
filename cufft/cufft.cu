#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cufft.h"
#include <iostream>


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

int main()
{
	const int Length = 10;
	cufftComplex *signalIn,*signalOut;
	signalIn = (cufftComplex *)malloc(Length * sizeof(cufftComplex));
	signalOut= (cufftComplex *)malloc(Length * sizeof(cufftComplex));
	for (int i = 0; i < Length; ++i)
	{
		signalIn[i].x = (float)i;
		signalIn[i].y = 2.0f;
	}

	printf("data to be tranforded:\n");
	for (int i = 0; i < Length; ++i)
		printf("%f+j%f\n", signalIn[i].x, signalIn[i].y);
	printf("\n");

	//alloc memory in device
	cufftComplex *dSignalIn,*dSignalOut;
	cudaMalloc((void **)&dSignalIn, Length * sizeof(cufftComplex));
	cudaMalloc((void **)&dSignalOut, Length * sizeof(cufftComplex));

	//copy memory from to host
	cudaMemcpy(dSignalIn, signalIn, Length * sizeof(cufftComplex), cudaMemcpyHostToDevice);


	//create handle of forward transform and inverse transform
	cufftHandle fftHandle, fftInverseHandle;
	fftStatus(cufftPlan1d(&fftHandle, Length, CUFFT_C2C, 1));
	fftStatus(cufftPlan1d(&fftInverseHandle, Length, CUFFT_C2C, 1));

	cufftExecC2C(fftHandle, dSignalIn, dSignalOut, CUFFT_FORWARD);
	//copy memory from device to host
	checkError(cudaMemcpy(signalOut, dSignalOut, Length * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

	printf("\ntransform result:\n");
	for (int i = 0; i < Length; ++i)
		printf("%f+j%f\n", signalOut[i].x, signalOut[i].y);
	printf("\n");

	//inverse transform
	cufftExecC2C(fftInverseHandle, dSignalOut, dSignalIn, CUFFT_INVERSE);
	checkError(cudaMemcpy(signalIn, dSignalIn, Length * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

	printf("\ninverse transform result:\n");
	for (int i = 0; i < Length; ++i)
		printf("%f+j%f\n", signalIn[i].x, signalIn[i].y);
	printf("\n");

	cudaFree(signalIn);
	cudaFree(signalOut);
	cufftDestroy(fftHandle);
	cufftDestroy(fftInverseHandle);
	free(signalIn);
	free(signalOut);

	return 0;
}
