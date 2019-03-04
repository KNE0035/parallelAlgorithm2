#include "Runner2.cuh"
#include "cudaUtils.cuh"
#include <cudaDefs.h>
#include <device_functions.h>

#define BLOCKSIZE_x 8
#define BLOCKSIZE_y 8


__host__ __device__ int iDivUp(int a, int b) { return ((a % b) != 0) ? (a / b + 1) : (a / b); }


void printMatrix(unsigned int* matrix, size_t nCols, size_t nRows) {
	for (int i = 0; i < nRows; i++) {
		for (int j = 0; j < nCols; j++) {
			printf("%d ", matrix[j + i * nCols]);
		}
		printf("\n");
	}
}

__global__ void fillMatrix(unsigned int* matrix, const size_t pitch, const size_t nCols, const size_t nRows)
{
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;

	while(idx < nRows && idy < nCols) {
		unsigned int value = idx + idy * nRows;
		*((char*)matrix + sizeof(unsigned int) * idx + idy * pitch) = value;
		idx += gridDim.x * blockDim.x;
		idy += gridDim.y * blockDim.y;
	}
}

void createMatrixOnDevice(const size_t nCols,const size_t nRows) {
	unsigned int* dMatrix;
	unsigned int* hMatrix = new unsigned int[nCols * nRows];
	size_t pitch;

	gpuErrorCheck(cudaMallocPitch((void**)&dMatrix, &pitch, nRows * sizeof(unsigned int), nCols));
	

	dim3 gridSize((nRows - 1) / BLOCKSIZE_y + 1, (nCols - 1) / BLOCKSIZE_x + 1);
	dim3 blockSize(BLOCKSIZE_y, BLOCKSIZE_x);
	
	fillMatrix << <gridSize, blockSize >> > (dMatrix, pitch, nCols, nRows);
	
	gpuErrorCheck(cudaDeviceSynchronize());
	gpuErrorCheck(cudaMemcpy2D(hMatrix, nRows * sizeof(unsigned int), dMatrix, pitch, nRows * sizeof(unsigned int), nCols, cudaMemcpyDeviceToHost));

	printMatrix(hMatrix, nCols, nRows);

	delete hMatrix;
	cudaFree(dMatrix);
}