#include <cudaDefs.h>

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

void testAdding2Vectors();
void testAddingNVectors();


__global__ void add2VectorsKernel(double *resultVector, const double *vectorA, const double *vectorB)
{
	int i = threadIdx.x;
	resultVector[i] = vectorA[i] + vectorB[i];
}

__global__ void addNVectorsKernel(double *resultVector, const double *vectorsArray, const int* n)
{
	int i = threadIdx.x;

	for (int j = 0; j < (*n); j++) {
		resultVector[i] += vectorsArray[i + j * (*n)];
	}
}

int main(int argc, char *argv[])
{
	initializeCUDA(deviceProp);
	testAdding2Vectors();
	testAddingNVectors();
}

void testAdding2Vectors () {
	const int dimmensions = 3;

	const double vectorA[dimmensions] = { 2, 2, 3 };
	const double vectorB[dimmensions] = { 4, 5, 6 };
	double resultVector[dimmensions] = { 0 };

	double* devA = 0;
	double* devB = 0;
	double* devResult = 0;

	error = cudaMalloc((void**)&devA, dimmensions * sizeof(double));
	if (error != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	error = cudaMalloc((void**)&devB, dimmensions * sizeof(double));
	if (error != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	error = cudaMalloc((void**)&devResult, dimmensions * sizeof(double));
	if (error != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}


	error = cudaMemcpy(devA, vectorA, dimmensions * sizeof(double), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	error = cudaMemcpy(devB, vectorB, dimmensions * sizeof(double), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	add2VectorsKernel <<<1, dimmensions >>> (devResult, devA, devB);

	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(error));
	}

	error = cudaDeviceSynchronize();
	if (error != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", error);
	}

	error = cudaMemcpy(resultVector, devResult, dimmensions * sizeof(double), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	printf("Result vector: ");

	for (int i = 0; i < dimmensions; i++) {
		printf(" %f, ", resultVector[i]);
	}

	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devResult);
}

void testAddingNVectors() {
	double* devVectors = 0;
	double* devResult = 0;
	int* devNumberOfVectors = 0;
	
	const int dimmensions = 3;
	const int numberOfVectors = 3;

	double resultVector[dimmensions] = { 0 };
	double* vectorsArray = new double[dimmensions * numberOfVectors];
	std::fill(vectorsArray, vectorsArray + (dimmensions * numberOfVectors), 3);

	error = cudaMalloc((void**)&devVectors, dimmensions * numberOfVectors * sizeof(double));
	if (error != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	error = cudaMalloc((void**)&devNumberOfVectors, sizeof(int));
	if (error != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	error = cudaMalloc((void**)&devResult, dimmensions * sizeof(double));
	if (error != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	error = cudaMemcpy(devVectors, vectorsArray, dimmensions * numberOfVectors * sizeof(double), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	error = cudaMemcpy(devNumberOfVectors, &numberOfVectors, sizeof(int), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	addNVectorsKernel <<<1, dimmensions >>> (devResult, devVectors, devNumberOfVectors);

	error = cudaDeviceSynchronize();
	if (error != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", error);
	}

	error = cudaMemcpy(resultVector, devResult, dimmensions * sizeof(double), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	printf("Result vector: ");

	for (int i = 0; i < dimmensions; i++) {
		printf(" %f, ", resultVector[i]);
	}

	cudaFree(devVectors);
	cudaFree(devResult);
	cudaFree(devNumberOfVectors);
}