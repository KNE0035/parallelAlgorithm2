#include <cudaDefs.h>
#include <time.h>
#include <math.h>
#include <random>
#include <device_functions.h>

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

using namespace std;

texture<float3, cudaTextureType1D, cudaReadModeElementType> pointTex;

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
	initializeCUDA(deviceProp);
	float3* points = createPointsData(length);
	printData(points, length);
	createSrcTexure(points, length);
	
}
