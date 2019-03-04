#include "Runner.cuh"
#include "cudaUtils.cuh"

#include <cudaDefs.h>


cudaDeviceProp deviceProp = cudaDeviceProp();


constexpr unsigned int THREADS_PER_BLOCK = 256;

void testAdding2Vectors();
void testAddingNVectors();


__global__ void add2VectorsKernel(float *resultVector, const float *vectorA, const float *vectorB, const unsigned int lenght)
{
	unsigned int offset = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
	
	while (offset < lenght) {
		resultVector[offset] = vectorA[offset] + vectorB[offset];
		offset += gridDim.x * THREADS_PER_BLOCK;
	}
}

__global__ void addNVectorsKernel(double *resultVector, const double *vectorsArray, const unsigned int nOVectors, const unsigned int vectorsLength)
{
	unsigned int offset = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
	while (offset < vectorsLength) {
		for (unsigned int j = 0; j < nOVectors; j++) {
			resultVector[offset] += vectorsArray[offset + j * nOVectors];
		}
		offset += gridDim.x * THREADS_PER_BLOCK;
	}
}

void testAdding2Vectors () {
	initializeCUDA(deviceProp);		cudaEvent_t startEvent, stopEvent;	float elapsedTime;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);

	const int dimmensions = 10;

	float* vectorA= new float[dimmensions];
	float* vectorB = new float[dimmensions];
	float* resultVector = new float[dimmensions];

	std::fill(vectorA, vectorA + (dimmensions), 1000);
	std::fill(vectorB, vectorB + (dimmensions), 1000);

	float* devA = 0;
	float* devB = 0;
	float* devResult = 0;

	gpuErrorCheck(cudaMalloc((void**)&devA, dimmensions * sizeof(float)));
	gpuErrorCheck(cudaMalloc((void**)&devB, dimmensions * sizeof(float)));
	gpuErrorCheck(cudaMalloc((void**)&devResult, dimmensions * sizeof(float)));

	gpuErrorCheck(cudaMemcpy(devA, vectorA, dimmensions * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(devB, vectorB, dimmensions * sizeof(float), cudaMemcpyHostToDevice));

	dim3 DimGrid((dimmensions - 1) / THREADS_PER_BLOCK + 1);
	dim3 DimBlock(THREADS_PER_BLOCK);

	cudaEventRecord(startEvent, 0);

	add2VectorsKernel <<<DimGrid, DimBlock>>> (devResult, devA, devB, dimmensions);

	cudaEventRecord(stopEvent, 0);
	gpuErrorCheck(cudaDeviceSynchronize());
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);


	gpuErrorCheck(cudaMemcpy(resultVector, devResult, dimmensions * sizeof(float), cudaMemcpyDeviceToHost));
	printf("\n time: %f \n\n", elapsedTime);
	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);

	/*printf("Result vector: ");
	for (int i = 0; i < dimmensions; i++) {
		printf(" %f, ", resultVector[i]);
	}*/

	delete vectorA;
	delete vectorB;
	delete resultVector;

	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devResult);
}

void testAddingNVectors() {
	double* devVectors = 0;
	double* devResult = 0;

	const int dimmensions = 3;
	const int numberOfVectors = 3;

	double resultVector[dimmensions] = { 0 };
	double* vectorsArray = new double[dimmensions * numberOfVectors];
	std::fill(vectorsArray, vectorsArray + (dimmensions * numberOfVectors), 3);

	gpuErrorCheck(cudaMalloc((void**)&devVectors, dimmensions * numberOfVectors * sizeof(double)));
	gpuErrorCheck(cudaMalloc((void**)&devResult, dimmensions * sizeof(double)));

	gpuErrorCheck(cudaMemcpy(devVectors, vectorsArray, dimmensions * numberOfVectors * sizeof(double), cudaMemcpyHostToDevice));

	dim3 DimGrid((dimmensions - 1) / THREADS_PER_BLOCK + 1);
	dim3 DimBlock(THREADS_PER_BLOCK);
	addNVectorsKernel <<<DimGrid, DimBlock>>> (devResult, devVectors, dimmensions, numberOfVectors);

	gpuErrorCheck(cudaDeviceSynchronize());

	gpuErrorCheck(cudaMemcpy(resultVector, devResult, dimmensions * sizeof(double), cudaMemcpyDeviceToHost));

	printf("Result vector: ");
	for (int i = 0; i < dimmensions; i++) {
		printf(" %f, ", resultVector[i]);
	}

	delete vectorsArray;

	cudaFree(devVectors);
	cudaFree(devResult);
}