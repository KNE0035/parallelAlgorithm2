#include <cudaDefs.h>
#include <time.h>
#include <math.h>
#include <random>
#include <device_functions.h>

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

using namespace std;

texture<float3, cudaTextureType1D, cudaReadModeElementType> pointTex;
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

__global__ void kMeansKernel(const int lenght, float3** pointGroups, float* sumOfSquares)
{
}

void createSrcTexure(float3* points, const unsigned int length) {
	cudaChannelFormatDesc texChannelDesc;
	
	texChannelDesc = cudaCreateChannelDesc(32, 32, 32, 0, cudaChannelFormatKindFloat);
	pointTex.normalized = false;
	pointTex.filterMode = cudaFilterModePoint;
	pointTex.addressMode[0] = cudaAddressModeBorder;
	cudaBindTexture(0, &pointTex, points, &texChannelDesc, length);
}

int main(int argc, char *argv[])
{
	const unsigned int length = 10;
	const unsigned int nOIterations = 100;
	const unsigned k = 10;

	initializeCUDA(deviceProp);
	float3** dPointGroupsResult;
	float minimumSumOfSquares = FLT_MAX;
	float* dSumOfSquares;
	float hSumOfSquares;
	float3* points = createPointsData(length);


	printData(points, length);
	createSrcTexure(points, length);
	
	cudaMalloc(&dSumOfSquares, sizeof(float));

	for (unsigned int i = 0; i < nOIterations; i++) {
		kmeansKernelSettings.dimBlock = dim3(BLOCK_DIM, 1, 1);
		kmeansKernelSettings.blockSize = BLOCK_DIM;
		kmeansKernelSettings.dimGrid = dim3((length + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
		
		kMeansKernel << <kmeansKernelSettings.dimGrid, kmeansKernelSettings.dimBlock >> > (length, dPointGroupsResult, sumOfSquares);
		cudaMemcpy(&hSumOfSquares, dSumOfSquares, sizeof(float), cudaMemcpyDeviceToHost);

		if (minimumSumOfSquares > hSumOfSquares) {
			minimumSumOfSquares = hSumOfSquares;

		}

	}
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

