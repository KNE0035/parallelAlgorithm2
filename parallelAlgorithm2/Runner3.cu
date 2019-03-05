#include "Runner3.cuh"
#include <cudaDefs.h>
#include <time.h>
#include <math.h>
#include <random>
#include <device_functions.h>

//WARNING!!! Do not change TPB and NO_FORCES for this demo !!!
constexpr unsigned int TPB = 128;
constexpr unsigned int NO_FORCES = 256;
constexpr unsigned int NO_RAIN_DROPS = 1 << 20;

constexpr unsigned int MEM_BLOCKS_PER_THREAD_BLOCK = 8;

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

using namespace std;

__host__ float3 *createData(const unsigned int length)
{
	random_device rd;
	mt19937_64 mt(rd());
	uniform_int_distribution<float> dist(0.0f, 1.0f);

	float3 *data = static_cast<float3*>(::operator new(sizeof(float3)* length));

	for (unsigned int i = 0; i < length; i++) {
		//data[i] = make_float3(dist(mt), dist(mt), dist(mt));
		data[i] = make_float3(1.0f, 1.0f, 1.0f);
	}

	//TODO: Generate float3 vectors. You can use 'make_float3' method.
	return data;
}

void printData(const float3 *data, const unsigned int length)
{
	if (data == 0) return;
	const float3 *ptr = data;
	for (unsigned int i = 0; i<length; i++, ptr++)
	{
		printf("%5.2f %5.2f %5.2f ", ptr->x, ptr->y, ptr->z);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Sums the forces to get the final one using parallel reduction. 
/// 		    WARNING!!! The method was written to meet input requirements of our example, i.e. 128 threads and 256 forces  </summary>
/// <param name="dForces">	  	The forces. </param>
/// <param name="noForces">   	The number of forces. </param>
/// <param name="dFinalForce">	[in,out] If non-null, the final force. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////
extern void reduce(const float3 * __restrict__ dForces, const unsigned int noForces, float3* __restrict__ dFinalForce)
{
	
	
	__shared__ float3 sForces[TPB];		//SEE THE WARNING MESSAGE !!!
	unsigned int tid = threadIdx.x;
	unsigned int next = TPB;			//SEE THE WARNING MESSAGE !!!
	
	float3 * src = &sForces[tid];
	float3 * src2 = (float3*)&sForces[tid + next];

	#pragma unroll
	for (unsigned int s = (blockDim.x >> 1); s>32; s >>= 1)
	{
		if (tid >= s) return;

		src->x += src2->x;
		src->y += src2->y;
		src->z += src2->z;
		src2 = src + s;
		__syncthreads();
	}

	volatile float3 *vsrc = &sForces[tid];

	if (tid < 32)
	{
		vsrc[tid].x += vsrc[tid + 32].x;
		vsrc[tid].y += vsrc[tid + 32].y;
		vsrc[tid].z += vsrc[tid + 32].z;

		vsrc[tid].x += vsrc[tid + 16].x;
		vsrc[tid].y += vsrc[tid + 16].y;
		vsrc[tid].z += vsrc[tid + 16].z;

		vsrc[tid].x += vsrc[tid + 8].x;
		vsrc[tid].y += vsrc[tid + 8].y;
		vsrc[tid].z += vsrc[tid + 8].z;

		vsrc[tid].x += vsrc[tid + 4].x;
		vsrc[tid].y += vsrc[tid + 4].y;
		vsrc[tid].z += vsrc[tid + 4].z;

		vsrc[tid].x += vsrc[tid + 2].x;
		vsrc[tid].y += vsrc[tid + 2].y;
		vsrc[tid].z += vsrc[tid + 2].z;

		vsrc[tid].x += vsrc[tid + 1].x;
		vsrc[tid].y += vsrc[tid + 1].y;
		vsrc[tid].z += vsrc[tid + 1].z;
	}

	dFinalForce->x = vsrc->x;
	dFinalForce->y = vsrc->y;
	dFinalForce->z = vsrc->z;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Adds the FinalForce to every Rain drops position. </summary>
/// <param name="dFinalForce">	The final force. </param>
/// <param name="noRainDrops">	The number of rain drops. </param>
/// <param name="dRainDrops"> 	[in,out] If non-null, the rain drops positions. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void add(const float3* __restrict__ dFinalForce, const unsigned int noRainDrops, float3* __restrict__ dRainDrops)
{
	//TODO: Add the FinalForce to every Rain drops position.
}


int particleSystemSimulation()
{
	initializeCUDA(deviceProp);

	cudaEvent_t startEvent, stopEvent;
	float elapsedTime;

	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	cudaEventRecord(startEvent, 0);

	float3 *hForces = createData(NO_FORCES);
	float3 *hDrops = createData(NO_RAIN_DROPS);

	float3 *dForces = nullptr;
	float3 *dDrops = nullptr;
	float3 *dFinalForce = nullptr;

	error = cudaMalloc((void**)&dForces, NO_FORCES * sizeof(float3));
	error = cudaMemcpy(dForces, hForces, NO_FORCES * sizeof(float3), cudaMemcpyHostToDevice);

	error = cudaMalloc((void**)&dDrops, NO_RAIN_DROPS * sizeof(float3));
	error = cudaMemcpy(dDrops, hDrops, NO_RAIN_DROPS * sizeof(float3), cudaMemcpyHostToDevice);

	error = cudaMalloc((void**)&dFinalForce, sizeof(float3));

	KernelSetting ksReduce;

	//TODO: ... Set ksReduce


	KernelSetting ksAdd;
	//TODO: ... Set ksAdd
	
	for (unsigned int i = 0; i<1000; i++)
	{
		reduce<<<ksReduce.dimGrid, ksReduce.dimBlock>>>(dForces, NO_FORCES, dFinalForce);
		add<<<ksAdd.dimGrid, ksAdd.dimBlock>>>(dFinalForce, NO_RAIN_DROPS, dDrops);
	}

	checkDeviceMatrix<float>((float*)dFinalForce, sizeof(float3), 1, 3, "%5.2f ", "Final force");
	// checkDeviceMatrix<float>((float*)dDrops, sizeof(float3), NO_RAIN_DROPS, 3, "%5.2f ", "Final Rain Drops");

	if (hForces)
		free(hForces);
	if (hDrops)
		free(hDrops);

	cudaFree(dForces);
	cudaFree(dDrops);

	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);

	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);

	printf("Time to get device properties: %f ms", elapsedTime);
}
