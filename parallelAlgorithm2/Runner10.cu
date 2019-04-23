#include <cudaDefs.h>
#include <time.h>
#include <math.h>

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

const unsigned int N = 1 << 20;
const unsigned int MEMSIZE = N * sizeof(unsigned int);
const unsigned int NO_LOOPS = 100;
const unsigned int THREAD_PER_BLOCK = 256;
const unsigned int GRID_SIZE = (N + THREAD_PER_BLOCK - 1)/THREAD_PER_BLOCK;

void fillData(unsigned int *data, const unsigned int length)
{
	//srand(time(0));
	for (unsigned int i=0; i<length; i++)
	{
		//data[i]= rand();
		data[i]= 1;
	}
}

void printData(const unsigned int *data, const unsigned int length)
{
	if (data ==0) return;
	for (unsigned int i=0; i<length; i++)
	{
		printf("%u ", data[i]);
	}
}


__global__ void kernel(const unsigned int *a, const unsigned int *b, const unsigned int length, unsigned int *c)
{
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	//TODO:  thread block loop
	if (tid < length)
	{
		c[tid] = a[tid] + b[tid];
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Tests 1. - single stream, async calling </summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
void test1()
{
	unsigned int *a, *b, *c;
	unsigned int *da, *db, *dc;

	// paged-locked allocation
	cudaHostAlloc((void**)&a, NO_LOOPS * MEMSIZE,cudaHostAllocDefault);
	cudaHostAlloc((void**)&b, NO_LOOPS * MEMSIZE,cudaHostAllocDefault);
	cudaHostAlloc((void**)&c, NO_LOOPS * MEMSIZE,cudaHostAllocDefault);

	fillData(a, NO_LOOPS * N);
	fillData(b, NO_LOOPS * N);

	// Data chunks on GPU
	cudaHostAlloc( (void**)&da, MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc( (void**)&db, MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc( (void**)&dc, MEMSIZE, cudaHostAllocDefault);

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	unsigned int dataOffset = 0;

	cudaEvent_t startEvent, stopEvent;
	float elapsedTime;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	cudaEventRecord(startEvent, 0);

	for(int i=0; i < NO_LOOPS; i++)
	{
		

		cudaMemcpyAsync(da, &a[dataOffset], MEMSIZE, cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(db, &b[dataOffset], MEMSIZE, cudaMemcpyHostToDevice, stream);

		kernel <<< GRID_SIZE, THREAD_PER_BLOCK,0,stream>>>(da, db, N, dc);
		cudaMemcpyAsync(&c[dataOffset], dc, MEMSIZE, cudaMemcpyHostToDevice, stream);
		dataOffset += N;
	}

	//TODO: Synchonize stream
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);

	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	printf("Test time: %f ms\n", elapsedTime);

	printData(c, 100);
	
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Tests 2. - two streams - depth first approach </summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
void test2()
{
	unsigned int *a, *b, *c;
	unsigned int *da1, *db1, *dc1, *da2, *db2, *dc2;

	// paged-locked allocation
	cudaHostAlloc((void**)&a, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&b, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&c, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);

	fillData(a, NO_LOOPS * N);
	fillData(b, NO_LOOPS * N);

	// Data chunks on GPU
	cudaMalloc((void**)&da1, MEMSIZE);
	cudaMalloc((void**)&db1, MEMSIZE);
	cudaMalloc((void**)&dc1, MEMSIZE);

	cudaMalloc((void**)&da2, MEMSIZE);
	cudaMalloc((void**)&db2, MEMSIZE);
	cudaMalloc((void**)&dc2, MEMSIZE);

	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	unsigned int dataOffset = 0;

	cudaEvent_t startEvent, stopEvent;
	float elapsedTime;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	cudaEventRecord(startEvent, 0);

	for (int i = 0; i < NO_LOOPS; i += 2)
	{

		cudaMemcpyAsync(da1, &a[dataOffset], MEMSIZE, cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(db1, &b[dataOffset], MEMSIZE, cudaMemcpyHostToDevice, stream1);
		kernel << < GRID_SIZE, THREAD_PER_BLOCK, 0, stream1 >> >(da1, db1, N, dc1);
		cudaMemcpyAsync(&c[dataOffset], dc1, MEMSIZE, cudaMemcpyHostToDevice, stream1);

		dataOffset += N;

		cudaMemcpyAsync(da2, &a[dataOffset], MEMSIZE, cudaMemcpyHostToDevice, stream2);
		cudaMemcpyAsync(db2, &b[dataOffset], MEMSIZE, cudaMemcpyHostToDevice, stream2);
		kernel << < GRID_SIZE, THREAD_PER_BLOCK, 0, stream2 >> >(da1, db1, N, dc1);
		cudaMemcpyAsync(&c[dataOffset], dc2, MEMSIZE, cudaMemcpyHostToDevice, stream2);
		
		dataOffset += N;
	}

	//TODO: Synchonize stream
	cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream2);

	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);

	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	printf("Test time: %f ms\n", elapsedTime);

	printData(c, 100);

	cudaFree(da1);
	cudaFree(db1);
	cudaFree(dc1);

	cudaFree(da2);
	cudaFree(db2);
	cudaFree(dc2);

	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Tests 3. - two streams - breadth first approach</summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
void test3()
{
	//TODO: reuse the source code of above mentioned method test1()
}


int main(int argc, char *argv[])
{
	initializeCUDA(deviceProp);

	test1();
	test2();
	test3();

	return 0;
}
