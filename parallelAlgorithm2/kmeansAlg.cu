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

ClusteredPoint* calculateKMeans(float3* points, const unsigned int k, const unsigned int length);

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();
using namespace std;
constexpr unsigned int N = 1000000;

//texture<float4, cudaTextureType1D, cudaReadModeElementType> pointTex;
KernelSetting kmeansDistributeKernelSettings;

#define BLOCK_DIM 1024

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
	for (unsigned int i =0; i < length; i++)
	{
		float vectorSize = sqrtf(dot(points[i].point, points[i].point));
		printf("%5.2f %5.2f %5.2f, cluster: %d (vectorSize: %f)\n", points[i].point.x, points[i].point.y, points[i].point.z, points[i].cluster, vectorSize);
	}
}

__global__ void kMeansInitClusteredPoints(const unsigned int length, float3* dPoints, ClusteredPoint* dClusteredPoints)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= length) {
		return;
	}

	dClusteredPoints[idx].cluster = -1;
	dClusteredPoints[idx].point = dPoints[idx];
}

__global__ void kMeansIsGroupsIdentic(const unsigned int length, ClusteredPoint* dClusteredPointsPrev, ClusteredPoint* dClusteredPointsNext, bool* identic)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= length) {
		return;
	}
	if (dClusteredPointsPrev[idx].cluster != dClusteredPointsNext[idx].cluster) {
		if (*identic) {
			*identic = false;
		}
	}
}


__global__ void kMeansDistribute(const unsigned int length, const unsigned int k, float3* centroids, unsigned int* clusterLengthArray, float3* sumByCentroids, ClusteredPoint* clusteredPoints)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= length) {
		return;
	}

	unsigned int minDistanceCentroidNumber;
	float minSumOfSquares = FLT_MAX;
	float3 point = clusteredPoints[idx].point;
	//calculating min distance to centroids
	for (unsigned int i = 0; i < k; i++)
	{
		float sumOfSquares = 0;

		float differenceX = point.x - centroids[i].x;
		float differenceY = point.y - centroids[i].y;
		float differenceZ = point.z - centroids[i].z;

		sumOfSquares += differenceX * differenceX;
		sumOfSquares += differenceY * differenceY;
		sumOfSquares += differenceZ * differenceZ;

		if (minSumOfSquares > sumOfSquares) {
			minSumOfSquares = sumOfSquares;
			minDistanceCentroidNumber = i;
		}
	}
	clusteredPoints[idx].point = point;
	clusteredPoints[idx].cluster = minDistanceCentroidNumber;

	atomicAdd(&(sumByCentroids[clusteredPoints[idx].cluster].x), clusteredPoints[idx].point.x);
	atomicAdd(&(sumByCentroids[clusteredPoints[idx].cluster].y), clusteredPoints[idx].point.y);
	atomicAdd(&(sumByCentroids[clusteredPoints[idx].cluster].z), clusteredPoints[idx].point.z);

	atomicAdd(&clusterLengthArray[clusteredPoints[idx].cluster], 1);
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
	const unsigned k = 3;

	initializeCUDA(deviceProp);
	float3 pointsTest[] = { make_float3(0,1,4),
					 make_float3(0,2,6) ,
					 make_float3(5,1,1) ,
					 make_float3(7,3,4) ,
					 make_float3(70,31, 42) 
					};
	
	float3* points = createPointsData(N);
	ClusteredPoint* result;

	//printData(points, length);
	cout << endl;

	cudaEvent_t startEvent, stopEvent;
	float elapsedTime;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	cudaEventRecord(startEvent, 0);
	result = calculateKMeans(points, k, N);

	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	printf("Test time: %f ms\n", elapsedTime);
	//printResult(result, N);
}

ClusteredPoint* calculateKMeans(float3* points, const unsigned int k, const unsigned int length) {
	kmeansDistributeKernelSettings.dimBlock = dim3(BLOCK_DIM, 1, 1);
	kmeansDistributeKernelSettings.blockSize = BLOCK_DIM;
	kmeansDistributeKernelSettings.dimGrid = dim3(getNumberOfParts(length, BLOCK_DIM), 1, 1);
	
	bool* dIdentic;
	bool* hIdentic = new bool();
	*hIdentic = true;

	ClusteredPoint* dClusteredPointsPrev;
	ClusteredPoint* dClusteredPointsNext;
	float3* dPoints;

	unsigned int* dClusterLengthArray;
	float3* dSumByCentroids;
	
	float3* dCentroids;
	float3* hCentroids = new float3[k];

	float3* hSumByCentroids = new float3[k];
	unsigned int* hClusterLengthArray = new unsigned int[k];

	float hSumOfSquares;
	ClusteredPoint* hclusteredPointsPrev = new ClusteredPoint[length];
	ClusteredPoint* hclusteredPointsNext = new ClusteredPoint[length];

	for (int i = 0; i < k; i++) {
		hSumByCentroids[i] = make_float3(0.0f, 0.0f, 0.0f);;
		hClusterLengthArray[i] = 0;
		hCentroids[i] = points[i];
	}

	gpuErrorCheck(cudaMalloc((void**)&dClusteredPointsPrev, length * sizeof(ClusteredPoint)));
	gpuErrorCheck(cudaMalloc((void**)&dClusteredPointsNext, length * sizeof(ClusteredPoint)));
	
	gpuErrorCheck(cudaMalloc((void**)&dPoints, length * sizeof(float3)));
	gpuErrorCheck(cudaMemcpy(dPoints, points, length * sizeof(float3), cudaMemcpyHostToDevice));

	kMeansInitClusteredPoints << <kmeansDistributeKernelSettings.dimGrid, kmeansDistributeKernelSettings.dimBlock >> > (length, dPoints, dClusteredPointsPrev);
	gpuErrorCheck(cudaDeviceSynchronize());
	gpuErrorCheck(cudaMemcpy(dClusteredPointsNext, dClusteredPointsPrev, length * sizeof(ClusteredPoint), cudaMemcpyDeviceToDevice));

	gpuErrorCheck(cudaMalloc((void**)&dCentroids, k * sizeof(float3)));
	gpuErrorCheck(cudaMalloc((void**)&dClusterLengthArray, k * sizeof(unsigned int)));
	gpuErrorCheck(cudaMalloc((void**)&dSumByCentroids, k * sizeof(float3)));
	gpuErrorCheck(cudaMalloc((void**)&dIdentic, sizeof(bool)));
	
	gpuErrorCheck(cudaMemcpy(dCentroids, hCentroids, k * sizeof(float3), cudaMemcpyHostToDevice));

	do
	{
		*hIdentic = true;

		for (int i = 0; i < k; i++) {
			hSumByCentroids[i] = make_float3(0.0f, 0.0f, 0.0f);;
			hClusterLengthArray[i] = 0;
		}
		gpuErrorCheck(cudaMemcpy(dSumByCentroids, hSumByCentroids, k * sizeof(float3), cudaMemcpyHostToDevice));
		gpuErrorCheck(cudaMemcpy(dClusterLengthArray, hClusterLengthArray, k * sizeof(unsigned int), cudaMemcpyHostToDevice));

		gpuErrorCheck(cudaMemcpy(dIdentic, hIdentic, sizeof(bool), cudaMemcpyHostToDevice));

		kMeansDistribute << <kmeansDistributeKernelSettings.dimGrid, kmeansDistributeKernelSettings.dimBlock >> > (length, k, dCentroids, dClusterLengthArray, dSumByCentroids, dClusteredPointsPrev);
		gpuErrorCheck(cudaDeviceSynchronize());
		
		gpuErrorCheck(cudaMemcpy(hClusterLengthArray, dClusterLengthArray, k * sizeof(unsigned int), cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(hSumByCentroids, dSumByCentroids, k * sizeof(float3), cudaMemcpyDeviceToHost));

		for (int i = 0; i < k; i++) {
			hCentroids[i] = hSumByCentroids[i] / hClusterLengthArray[i];
		}

		for (int i = 0; i < k; i++) {
			hSumByCentroids[i] = make_float3(0.0f, 0.0f, 0.0f);;
			hClusterLengthArray[i] = 0;
		}
		gpuErrorCheck(cudaMemcpy(dSumByCentroids, hSumByCentroids, k * sizeof(float3), cudaMemcpyHostToDevice));
		gpuErrorCheck(cudaMemcpy(dClusterLengthArray, hClusterLengthArray, k * sizeof(unsigned int), cudaMemcpyHostToDevice));

		gpuErrorCheck(cudaMemcpy(dCentroids, hCentroids, k * sizeof(float3), cudaMemcpyHostToDevice));
		kMeansDistribute << <kmeansDistributeKernelSettings.dimGrid, kmeansDistributeKernelSettings.dimBlock >> > (length, k, dCentroids, dClusterLengthArray, dSumByCentroids, dClusteredPointsNext);
		gpuErrorCheck(cudaDeviceSynchronize());

		kMeansIsGroupsIdentic << <kmeansDistributeKernelSettings.dimGrid, kmeansDistributeKernelSettings.dimBlock >> > (length, dClusteredPointsPrev, dClusteredPointsNext, dIdentic);
		gpuErrorCheck(cudaDeviceSynchronize());
		gpuErrorCheck(cudaMemcpy(hIdentic, dIdentic, sizeof(bool), cudaMemcpyDeviceToHost));

		if (!hIdentic) {
			gpuErrorCheck(cudaMemcpy(dClusteredPointsPrev, dClusteredPointsNext, length * sizeof(ClusteredPoint), cudaMemcpyDeviceToDevice));
		}

		//gpuErrorCheck(cudaMemcpy(hclusteredPointsPrev, dClusteredPointsPrev, length * sizeof(ClusteredPoint), cudaMemcpyDeviceToHost));
		//gpuErrorCheck(cudaMemcpy(hclusteredPointsNext, dClusteredPointsNext, length * sizeof(ClusteredPoint), cudaMemcpyDeviceToHost));

		/*for (int i = 0; i < length; i++) {
			if (hclusteredPointsPrev[i].cluster != hclusteredPointsNext[i].cluster) {
				*hIdentic = false;
				gpuErrorCheck(cudaMemcpy(dClusteredPointsPrev, dClusteredPointsNext, length * sizeof(ClusteredPoint), cudaMemcpyDeviceToDevice));
				break;
			}
		}*/

		printf("sad");
	} while (!(*hIdentic));
	gpuErrorCheck(cudaMemcpy(hclusteredPointsPrev, dClusteredPointsPrev, length * sizeof(ClusteredPoint), cudaMemcpyDeviceToHost));

	return hclusteredPointsPrev;
}