// includes, cudaimageWidth
#include <cudaDefs.h>
#include <imageManager.h>

#include "imageKernels.cuh"

#define BLOCK_DIM 8

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

//Use the followings to store information about the input image that will be processed
unsigned char *dSrcImageData = 0;
unsigned int srcImageWidth;
unsigned int srcImageHeight;
unsigned int srcImageBPP;		//Bits Per Pxel = 8, 16, 24, or 32 bit
unsigned int srcImagePitch;

//Use the followings to access the input image through the texture reference
texture<float, 2, cudaReadModeElementType> srcTexRef;
cudaChannelFormatDesc srcTexCFD;
size_t srcTexPitch;
float *dSrcTexData = 0;

size_t dstTexPitch;
uchar3 *dstTexData = 0;

KernelSetting squareKs;
float *dOutputData = 0;

template<bool normalizeTexel>__global__ void floatHeighmapTextureToNormalmap(const unsigned int texWidth, const unsigned int texHeight, const unsigned int dstPitch, uchar3* dst)
{
	__constant__ unsigned int* SOBEL_X_FILTER = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
	__constant__ unsigned int* SOBEL_Y_FILTER = {1, 2, 1, 0, 0, 0, -1, -2, -1};

	unsigned int threadIdx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int threadIdy = threadIdx.y + blockIdx.y * blockDim.y;

	float x, y, z = 0;

	z = 0.5;

	unsigned int index = threadIdx + threadIdy * dstPitch;
	for (unsigned int i = 0; i < 3; i++) {
		for (unsigned int j = 0; j < 3; j++) {
			x += tex2D(dSrcTexData, threadIdx + (j - 1), threadIdy + (i - 1)) * SOBEL_X_FILTER[j + i * 3];
			y += tex2D(dSrcTexData, threadIdx + (j - 1), threadIdy + (i - 1)) * SOBEL_Y_FILTER[j + i * 3];
		}
	}

	x /= 9;
	y /= 9;

	float distance = sqrt(x * x + y * y + z * z);

	x /= distance;
	y /= distance;
	z /= distance;

	uchar3 texel;

	texel.x = (x + 1) * 127.5;
	texel.y = y * 255;
	texel.z = (z + 1) * 127.5;

	dst[index] = texel;
}

#pragma region STEP 1

//TASK:	Load the input image and store loaded data in DEVICE memory (dSrcImageData)

void loadSourceImage(const char* imageFileName)
{
	FreeImage_Initialise();
	FIBITMAP *tmp = ImageManager::GenericLoader(imageFileName, 0);

	srcImageWidth = FreeImage_GetWidth(tmp);
	srcImageHeight = FreeImage_GetHeight(tmp);
	srcImageBPP = FreeImage_GetBPP(tmp);
	srcImagePitch = FreeImage_GetPitch(tmp);		// FREEIMAGE aligns row data ... You have to use pitch instead of width

	cudaMalloc((void**)&dSrcImageData, srcImagePitch * srcImageHeight * srcImageBPP / 8);
	cudaMemcpy(dSrcImageData, FreeImage_GetBits(tmp), srcImagePitch * srcImageHeight * srcImageBPP / 8, cudaMemcpyHostToDevice);

	checkHostMatrix<unsigned char>(FreeImage_GetBits(tmp), srcImagePitch, srcImageHeight, srcImageWidth, "%hhu ", "Result of Linear Pitch Text");
	checkDeviceMatrix<unsigned char>(dSrcImageData, srcImagePitch, srcImageHeight, srcImageWidth, "%hhu ", "Result of Linear Pitch Text");

	FreeImage_Unload(tmp);
	FreeImage_DeInitialise();
}
#pragma endregion

#pragma region STEP 2

//TASK: Create a texture based on the source image. The input images can have variable BPP (Byte Per Pixel), but finally any such image will be converted into the floating-point texture using
//		the colorToFloat kernel.

void createSrcTexure()
{
	//TODO: Floating Point Texture Data
	//cudaMallocPitch( ...);

	//Converts custom image data to float and stores result in the float_pitch_linear_data
	switch (srcImageBPP)
	{
	case 8:  colorToFloat<8, 2> << <squareKs.dimGrid, squareKs.dimBlock >> >(dSrcImageData, srcImageWidth, srcImageHeight, srcImagePitch, srcTexPitch / sizeof(float), dSrcTexData); break;
	case 16: colorToFloat<16, 2> << <squareKs.dimGrid, squareKs.dimBlock >> >(dSrcImageData, srcImageWidth, srcImageHeight, srcImagePitch, srcTexPitch / sizeof(float), dSrcTexData); break;
	case 24: colorToFloat<24, 2> << <squareKs.dimGrid, squareKs.dimBlock >> >(dSrcImageData, srcImageWidth, srcImageHeight, srcImagePitch, srcTexPitch / sizeof(float), dSrcTexData); break;
	case 32: colorToFloat<32, 2> << <squareKs.dimGrid, squareKs.dimBlock >> >(dSrcImageData, srcImageWidth, srcImageHeight, srcImagePitch, srcTexPitch / sizeof(float), dSrcTexData); break;
	}
	checkDeviceMatrix<float>(dSrcTexData, srcTexPitch, srcImageHeight, srcImageWidth, "%6.1f ", "Result of Linear Pitch Text");

	//TODO: Texture settings

	//TODO: Bind texture
	//cudaBindTexture2D(...);
}
#pragma endregion

#pragma region STEP 3

//TASK:	Convert the input image into normal map. Use the binded texture (srcTexRef).

void createNormalMap()
{

	//TODO: Allocate Pitch memory dstTexData to store output texture
	cudaMallocPitch((void**)dSrcTexData, &dstTexPitch, srcImageWidth, srcImageHeight);

	//TODO: Call the kernel that creates the normal map.
	floatHeighmapTextureToNormalmap<true><<<squareKs.dimGrid, squareKs.dimBlock>>>();
	const unsigned int texWidth, const unsigned int texHeight, const unsigned int dstPitch, uchar3* dst

	check_data<uchar3>::checkDeviceMatrix(dstTexData, srcImageHeight, dstTexPitch / sizeof(uchar3), true, "%hhu %hhu %hhu %hhu | ", "Result of Linear Pitch Text");
}

#pragma endregion

#pragma region STEP 4

//TASK: Save output image (normal map)

void saveTexImage(const char* imageFileName)
{
	FreeImage_Initialise();

	FIBITMAP *tmp = FreeImage_Allocate(srcImageWidth, srcImageHeight, 24);
	unsigned int tmpPitch = srcImagePitch = FreeImage_GetPitch(tmp);		// FREEIMAGE align row data ... You have to use pitch instead of width
	checkCudaErrors(cudaMemcpy2D(FreeImage_GetBits(tmp), FreeImage_GetPitch(tmp), dstTexData, dstTexPitch, srcImageWidth * 3, srcImageHeight, cudaMemcpyDeviceToHost));
	//FreeImage_Save(FIF_PNG, tmp, imageFileName, 0);
	ImageManager::GenericWriter(tmp, imageFileName, FIF_PNG);
	FreeImage_Unload(tmp);
	FreeImage_DeInitialise();
}

#pragma endregion

void releaseMemory()
{
	cudaUnbindTexture(srcTexRef);
	if (dSrcImageData != 0)
		cudaFree(dSrcImageData);
	if (dSrcTexData != 0)
		cudaFree(dSrcTexData);
	if (dstTexData != 0)
		cudaFree(dstTexData);
	if (dOutputData)
		cudaFree(dOutputData);
}

int main(int argc, char *argv[])
{
	initializeCUDA(deviceProp);

	//STEP 1
	loadSourceImage("C:/Users\kne0035/dev/parallelAlgorithm2/parallelAlgorithm2/images/terrain3Kx3K.tif");

	//TODO: Setup the kernel settings
	squareKs.dimBlock = dim3(BLOCK_DIM, BLOCK_DIM, 1);
	squareKs.blockSize = BLOCK_DIM * BLOCK_DIM;
	squareKs.dimGrid = dim3((srcImageWidth + BLOCK_DIM - 1) / BLOCK_DIM, (srcImageHeight + BLOCK_DIM - 1) / BLOCK_DIM, 1);

	//Step 2 - create heighmap texture stored in the linear pitch memory
	createSrcTexure();

	//Step 3 - create the normal map
	createNormalMap();

	//Step 4 - save the normal map
	saveTexImage("C:/Users\kne0035/dev/parallelAlgorithm2/parallelAlgorithm2/images/noramlMap.bmp");

	releaseMemory();
}
