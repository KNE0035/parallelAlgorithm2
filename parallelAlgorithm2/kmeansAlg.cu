#include <cudaDefs.h>
#include <time.h>
#include <math.h>
#include <random>
#include <device_functions.h>
#include "cudaUtils.cuh"

struct ClusteredPoint {
	float3 point;
	int cluster;
};

__device__ void distributePointToCluster(unsigned int idx, float3* centroids, const unsigned int k, ClusteredPoint* clusteredPointsLastNew, const unsigned int pointsLastNewOffset);
__device__ void recalculateCentroids(unsigned int idx, float3* centroids, unsigned int* clusterLengthArray, const unsigned int k, ClusteredPoint* clusteredPointsLastNew, unsigned int length);
__device__ bool isLastAndNewClusteredPointsIdentic(unsigned int idx, ClusteredPoint* clusteredPointsLastNew, const unsigned int lenght);

__device__ bool isIdentic = true;

ClusteredPoint* calculateKMeans(float3* points, const unsigned int k, const unsigned int length);

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

using namespace std;

//texture<float4, cudaTextureType1D, cudaReadModeElementType> pointTex;
KernelSetting kmeansKernelSettings;

#define BLOCK_DIM 8

__host__ float3* createPointsData(const unsigned int length)
{
	random_device rd;
	mt19937_64 mt(rd());
	uniform_real_distribution<float> dist(0.0, 1000.0);

	float3 *data = static_cast<float3*>(::operator new(sizeof(float3) * length));

	for (unsigned int i = 0; i < length; i++) {
		data[i] = make_float3(dist(mt), dist(mt), dist(mt));
		//data[i] = make_float3(1.0f, 1.0f, 1.0f);
	}
	return data;
}

void printData(const float3 *data, const unsigned int length)
{
	if (data == 0) return;
	const float3 *ptr = data;
	for (unsigned int i = 0; i < length; i++, ptr++)
	{
		printf("%5.2f %5.2f %5.2f \n", ptr->x, ptr->y, ptr->z);
	}
}

void printResult(const ClusteredPoint *points, const unsigned int length)
{
	printf("result: \n");
	for (unsigned int i = 0; i < length; i++)
	{
		printf("%5.2f %5.2f %5.2f, cluster: %d \n", points[i].point.x, points[i].point.y, points[i].point.z, points[i].cluster);
	}
}

__global__ void kMeansKernel(const unsigned int lenght, const unsigned int k, float3* centroids, unsigned int* clusterLengthArray, ClusteredPoint* clusteredPointsLastNew, float* sumOfSquares)
{
	unsigned int idx = (threadIdx.x + blockIdx.x * blockDim.x);

	do {
		distributePointToCluster(idx, centroids, k, clusteredPointsLastNew, 0);
		__syncthreads();
		recalculateCentroids(idx, centroids, clusterLengthArray, k, clusteredPointsLastNew, lenght);
		__syncthreads();
		distributePointToCluster(idx, centroids, k, clusteredPointsLastNew, lenght);
		isIdentic = true;
		__syncthreads();
	} while (!isLastAndNewClusteredPointsIdentic(idx, clusteredPointsLastNew, lenght));
}

/*void createSrcTexure(float3* points, const unsigned int length) {
	cudaChannelFormatDesc texChannelDesc;
	
	gpuErrorCheck(cudaMalloc(&dPoints, length * sizeof(float3)));
	gpuErrorCheck(cudaMemcpy(dPoints, points, length * sizeof(float3), cudaMemcpyHostToDevice));

	texChannelDesc = cudaCreateChannelDesc(32, 32, 32, 0, cudaChannelFormatKindFloat);
	pointTex.normalized = false;
	pointTex.filterMode = cudaFilterModePoint;
	pointTex.addressMode[0] = cudaAddressModeBorder;
	cudaBindTexture(0, &pointTex, dPoints, &texChannelDesc, length);
}*/

int main(int argc, char *argv[])
{
	const unsigned int length = 10000000;
	const unsigned k = 5;

	initializeCUDA(deviceProp);
	float3* points = createPointsData(length);
	ClusteredPoint* result;

	//printData(points, length);
	cout << endl;

	result = calculateKMeans(points, k, length);
	//printResult(result, length);
}

ClusteredPoint* calculateKMeans(float3* points, const unsigned int k, const unsigned int length) {
	const unsigned int nOIterations = 100;
	float minimumSumOfSquares = FLT_MAX;
	float* dSumOfSquares;
	ClusteredPoint* dClusteredPointsLastNew;
	unsigned int* dClusterLengthArray;
	float3* dCentroids;

	float hSumOfSquares;
	ClusteredPoint* hclusteredPoints = new ClusteredPoint[length];

	for (int i = 0; i < length; i++) {
		hclusteredPoints[i].point = points[i];
		hclusteredPoints[i].cluster = -1;
	}

	
	gpuErrorCheck(cudaMalloc((void**)&dSumOfSquares, sizeof(float)));
	gpuErrorCheck(cudaMalloc((void**)&dClusteredPointsLastNew, 2 * length * sizeof(ClusteredPoint)));
	gpuErrorCheck(cudaMalloc((void**)&dCentroids, k * sizeof(float3)));
	gpuErrorCheck(cudaMalloc((void**)&dClusterLengthArray, k * sizeof(unsigned int)));
	
	gpuErrorCheck(cudaMemcpy(dCentroids, points, k * sizeof(float3), cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(dClusteredPointsLastNew, hclusteredPoints, length * sizeof(ClusteredPoint), cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(dClusteredPointsLastNew + length, dClusteredPointsLastNew, length * sizeof(ClusteredPoint), cudaMemcpyDeviceToDevice));

	kmeansKernelSettings.dimBlock = dim3(BLOCK_DIM, 1, 1);
	kmeansKernelSettings.blockSize = BLOCK_DIM;
	kmeansKernelSettings.dimGrid = dim3((length + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);

	kMeansKernel << <kmeansKernelSettings.dimGrid, kmeansKernelSettings.dimBlock >> > (length, k, dCentroids, dClusterLengthArray, dClusteredPointsLastNew, dSumOfSquares);
	//gpuErrorCheck(cudaMemcpy(&hSumOfSquares, dSumOfSquares, sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(hclusteredPoints, dClusteredPointsLastNew, length * sizeof(ClusteredPoint), cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaDeviceSynchronize());

	return hclusteredPoints;
}

__device__ void distributePointToCluster(unsigned int idx, float3* centroids, const unsigned int k, ClusteredPoint* clusteredPoints, const unsigned int pointOffset)
{
	unsigned int minDistanceCentroidNumber;
	float minDistance = FLT_MAX;
	float3 point = clusteredPoints[idx].point;

	//calculating min distance to centroids
	for (unsigned int i = 0; i < k; i++)
	{
		float distance;
		float sumOfSquares;

		float differenceX = point.x - centroids[i].x;
		float differenceY = point.y - centroids[i].y;
		float differenceZ = point.z - centroids[i].z;

		sumOfSquares = differenceX * differenceX;
		sumOfSquares += differenceY * differenceY;
		sumOfSquares += differenceZ * differenceZ;
		distance = sqrtf(sumOfSquares);

		if (minDistance > distance) {
			minDistance = distance;
			minDistanceCentroidNumber = i;
		}
	}

	ClusteredPoint clusteredPoint;
	clusteredPoint.point = point;
	clusteredPoint.cluster = minDistanceCentroidNumber;
	clusteredPoints[idx + pointOffset] = clusteredPoint;
}

__device__ void recalculateCentroids(unsigned int idx, float3* centroids, unsigned int* clusterLengthArray, const unsigned int k, ClusteredPoint* clusteredPoints, unsigned int length) {
	
	if (idx > 0) {
		return;
	}

	for (int i = 0; i < k; i++) {
		centroids[i] = make_float3(0.0f, 0.0f, 0.0f);
		clusterLengthArray[i] = 0;
	}
	
	for (int i = 0; i < length; i++) {
		centroids[clusteredPoints[i].cluster] = centroids[clusteredPoints[i].cluster] + clusteredPoints[i].point;
		clusterLengthArray[clusteredPoints[i].cluster]++;
	}
	
	for (int i = 0; i < k; i++) {
		centroids[i] = centroids[i] / clusterLengthArray[i];
	}


}

__device__ bool isLastAndNewClusteredPointsIdentic(unsigned int idx, ClusteredPoint* clusteredPointsLastNew, const unsigned int lenght) {

	if (clusteredPointsLastNew[idx].cluster != clusteredPointsLastNew[idx + lenght].cluster) {
		isIdentic = false;
	}
	__syncthreads();
	return isIdentic;
}

//postup:
//1. na hostu shuffle pole bodu (mozna na device? jde paralelizovat?)
//2. zavolat kernel pro vypocet kmeans
	//2.1 vybrat k centroidu
	//2.2 spocitat matici vzdalenosti od centroidú (rozradit body k nejblizsim centroidúm (vznik skupin))
	//2.3 prepocitat centroidy (teziste bodu v dane skupine s centroidem)
	//2.4 spocitat matici vzdalenosti od novych centroidú (rozradit body k nejblizsim centroidúm (vznik skupin))
	//2.5 porovnat mnoziny bodú if stejne: pokracuj else: -> 2.3
	//2.6 vypocitat sum of squares pres vsechny skupiny a ulozit do vystupu
//3. Pokud pocet volani kernelu pro vypocet kmeans < nOIterations ? -> 2. (hlidat si vysledek s nejmensim sum of squares)
//4. vypsat skupiny bodu

