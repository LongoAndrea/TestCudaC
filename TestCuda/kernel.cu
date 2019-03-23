#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string.h>

#include <stdio.h>

#include "CSha.h"

#define SIZE_X 17576
#define SIZE_Y 4


using std::string;
using std::endl;
using std::cout;

typedef unsigned char BYTE;

cudaError_t memAlloc(BYTE *matrix);

__global__ void getCombinations(BYTE *matrix) 
{
	for (int i = 0; i < SIZE_X; i++)
		for (int j = 0; j < SIZE_Y; j++)
			matrix[i + j * SIZE_X] = 1;
}

int main()
{
	BYTE host_matrix[SIZE_X * SIZE_Y];

	for (int i = 0; i < SIZE_X; i++)
		for (int j = 0; j < SIZE_Y; j++)
			host_matrix[i + j * SIZE_X] = 0;

	memAlloc(host_matrix);

	return 0;
}

cudaError_t memAlloc(BYTE *matrix) 
{
	BYTE *dev_matrix;

	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dev_matrix, SIZE_X * SIZE_Y * sizeof(BYTE));
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaMalloc failed!");
		cudaFree(dev_matrix);
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(dev_matrix, matrix, SIZE_X * SIZE_Y * sizeof(BYTE), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(dev_matrix);
		return cudaStatus;
	}


	getCombinations <<< 1, SIZE_Y >>> (dev_matrix);
	
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		cudaFree(dev_matrix);
		return cudaStatus;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		cudaFree(dev_matrix);
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(matrix, dev_matrix, SIZE_X * SIZE_Y * sizeof(BYTE), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(dev_matrix);
		return cudaStatus;
	}

	cudaFree(dev_matrix);
	return cudaStatus;
}