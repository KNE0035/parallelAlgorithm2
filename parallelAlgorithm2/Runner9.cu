#include "Runner3.cuh"
#include <cudaDefs.h>
#include <time.h>
#include <math.h>
#include <random>
#include <device_functions.h>

//WARNING!!! Do not change TPB and NO_FORCES for this demo !!!
constexpr unsigned int THREADS_PER_BLOCK = 512;
constexpr unsigned int LENGTH = 1 << 10;
constexpr unsigned int BLOCK_DIM = 8;
constexpr unsigned int NO_BLOCKS = 500;


cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

using namespace std;

__host__ int *createData(const unsigned int length)
{
	random_device rd;
	mt19937_64 mt(rd());
	uniform_int_distribution<int> dist(0, 10000);

	int *data = static_cast<int*>(::operator new(sizeof(int)* length));

	for (unsigned int i = 0; i < length; i++) {
		data[i] = dist(mt);
	}
	return data;
}

void printData(const float *data, const unsigned int length)
{
	if (data == 0) return;
	const float *ptr = data;
	for (unsigned int i = 0; i<length; i++, ptr++)
	{
		printf("%5.2f, ", data[i]);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Sums the forces to get the final one using parallel reduction. 
/// 		    WARNING!!! The method was written to meet input requirements of our example, i.e. 128 threads and 256 forces  </summary>
/// <param name="dForces">	  	The forces. </param>
/// <param name="noForces">   	The number of forces. </param>
/// <param name="dFinalForce">	[in,out] If non-null, the final force. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void getMax(int* data, const unsigned int length, int* result)
{
	unsigned int tIdOffset = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
	
	while (tIdOffset < length) {
		atomicMax(result, data[tIdOffset]);
		tIdOffset += gridDim.x * blockDim.x;
	}
}

int main()
{
	initializeCUDA(deviceProp);

	cudaEvent_t startEvent, stopEvent;
	float elapsedTime;

	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	cudaEventRecord(startEvent, 0);

	int* hData = createData(LENGTH);
	int* dData;
	int hMax = -1;
	int* dMax;

	error = cudaMalloc((void**)&dData, LENGTH * sizeof(int));
	error = cudaMalloc((void**)&dMax, sizeof(int));

	error = cudaMemcpy(dData, hData, LENGTH * sizeof(int), cudaMemcpyHostToDevice);
	error = cudaMemcpy(dMax, &hMax, sizeof(int), cudaMemcpyHostToDevice);

	KernelSetting ks;
	ks.dimBlock = dim3(THREADS_PER_BLOCK, 1, 1);
	ks.dimGrid = dim3(NO_BLOCKS, 1, 1);


	getMax << <ks.dimGrid, ks.dimBlock >> > (dData, LENGTH, dMax);

	error = cudaMemcpy(&hMax, dMax, sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);

	printf("Max: %d", hMax);

	cudaFree(dData);
	cudaFree(dMax);

	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);

	printf("Time to get max %f ms", elapsedTime);
}
