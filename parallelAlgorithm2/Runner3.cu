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
	uniform_real_distribution<float> dist(0.0, 1.0);

	float3 *data = static_cast<float3*>(::operator new(sizeof(float3)* length));

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
__global__ void reduce(const float3 * __restrict__ dForces, const unsigned int noForces, float3* __restrict__ dFinalForce)
{
	__shared__ float3 sForces[TPB];		//SEE THE WARNING MESSAGE !!!
	unsigned int tid = threadIdx.x;
	unsigned int next = TPB;			//SEE THE WARNING MESSAGE !!!
	float3 * src = &sForces[tid];
	*src = dForces[tid];
	float3 * src2 = (float3*)&dForces[tid + next];

	#pragma unroll MEM_BLOCKS_PER_THREAD_BLOCK
	for (next; next > 32; next >>= 1)
	{
		if (tid >= next) return;
		src->x += src2->x;
		src->y += src2->y;
		src->z += src2->z;
		//printf("%d\n", next);
		src2 = src + (next >> 1);
		__syncthreads();
	}
	if (tid >= next) return;

	volatile float3 *vsrc = &sForces[tid];
	volatile float3 *vsrc2;

	#pragma unroll MEM_BLOCKS_PER_THREAD_BLOCK
	for (next; next > 0; next >>= 1)
	{
		if (tid >= next) return;
		vsrc2 = vsrc + next;
		vsrc->x += vsrc2->x;
		vsrc->y += vsrc2->y;
		vsrc->z += vsrc2->z;
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
	float3 finalForce = *dFinalForce;
	unsigned int offset = blockDim.x;
	unsigned int index = MEM_BLOCKS_PER_THREAD_BLOCK * blockIdx.x * offset + threadIdx.x;

	float3* ptr = &dRainDrops[index];

	#pragma unroll MEM_BLOCKS_PER_THREAD_BLOCK
	for (unsigned int i = 0; i < MEM_BLOCKS_PER_THREAD_BLOCK; i++)
	{
		if (index >= noRainDrops) return;
		ptr->x += finalForce.x;
		ptr->y += finalForce.y;
		ptr->z += finalForce.z;
		ptr += offset;
		index += offset;
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

	ksReduce.dimBlock = dim3(TPB, 1, 1);
	ksReduce.dimGrid = dim3(1, 1, 1);

	KernelSetting ksAdd;
	ksAdd.dimBlock = dim3(TPB, 1, 1);
	ksAdd.dimGrid = dim3(getNumberOfParts(NO_RAIN_DROPS, TPB*MEM_BLOCKS_PER_THREAD_BLOCK), 1, 1);
	
	//printData(hForces, NO_FORCES);

	reduce << <ksReduce.dimGrid, ksReduce.dimBlock >> > (dForces, NO_FORCES, dFinalForce);
	for (unsigned int i = 0; i<1000; i++)
	{
		add<<<ksAdd.dimGrid, ksAdd.dimBlock>>>(dFinalForce, NO_RAIN_DROPS, dDrops);
	}

	checkDeviceMatrix<float>((float*)dFinalForce, sizeof(float3), 1, 3, "%5.2f ", "Final force");
	checkDeviceMatrix<float>((float*)dDrops, sizeof(float3), NO_RAIN_DROPS, 3, "%5.2f ", "Final Rain Drops");

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
