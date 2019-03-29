// includes, cuda
#include <cuda_runtime.h>

#include <cudaDefs.h>
#include <imageManager.h>


#include "imageKernels.cuh"

#define BLOCK_DIM 8

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

texture<float, 2, cudaReadModeElementType> texRef;
cudaChannelFormatDesc texChannelDesc;

unsigned char *dImageData = 0;
unsigned int imageWidth;
unsigned int imageHeight;
unsigned int imageBPP;		//Bits Per Pixel = 8, 16, 24, or 32 bit
unsigned int imagePitch;

size_t texPitch;
float *dLinearPitchTextureData = 0;
cudaArray *dArrayTextureData = 0;

uchar3 *dstTexData;

KernelSetting squareKs;;

float *dOutputData = 0;

__constant__  int SOBEL_X_FILTER[] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
__constant__  int SOBEL_Y_FILTER[] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

template<bool normalizeTexel>__global__ void floatHeighmapTextureToNormalmap(const unsigned int texWidth, const unsigned int texHeight, const unsigned int dstPitch, uchar3* dst)
{

	unsigned int col = (threadIdx.x + blockIdx.x * blockDim.x);
	unsigned int row = (threadIdx.y + blockIdx.y * blockDim.y);

	float x = 0, y = 0, z = 0;

	z = 0.5;

	unsigned int offset = col + row * (dstPitch / 3);
	
	for (unsigned int i = 0; i < 3; i++) {
		for (unsigned int j = 0; j < 3; j++) {
			float texel = tex2D(texRef, col + (j - 1), row + (i - 1));
			x += texel * SOBEL_X_FILTER[j + i * 3];
			y += texel * SOBEL_Y_FILTER[j + i * 3];
		}
	}
	x = x / 9;
	y = y / 9;
 
	if (normalizeTexel) {
		float distance = sqrt(x * x + y * y + z * z);
		x /= distance;
		y /= distance;
		z /= distance;
	}

	uchar3 rgbTexel;
	uchar3 bgrTexel;
	rgbTexel.x = (x + 1) * 127.5;
	rgbTexel.y = (y + 1) * 127.5;
	rgbTexel.z = z * 255;
	//printf("%u, %u, %u \n", texel.x, texel.y, texel.z);

	bgrTexel.x = rgbTexel.z;
	bgrTexel.y = rgbTexel.y;
	bgrTexel.z = rgbTexel.x;

	dst[offset] = rgbTexel;
}

#pragma region STEP 1

//TASK:	Load the input image and store loaded data in DEVICE memory (dSrcImageData)

void loadSourceImage(const char* imageFileName)
{
	FreeImage_Initialise();
	FIBITMAP *tmp = ImageManager::GenericLoader(imageFileName, 0);

	imageWidth = FreeImage_GetWidth(tmp);
	imageHeight = FreeImage_GetHeight(tmp);
	imageBPP = FreeImage_GetBPP(tmp);
	imagePitch = FreeImage_GetPitch(tmp);		// FREEIMAGE aligns row data ... You have to use pitch instead of width

	cudaMalloc((void**)&dImageData, imagePitch * imageHeight * imageBPP / 8);
	cudaMemcpy(dImageData, FreeImage_GetBits(tmp), imagePitch * imageHeight * imageBPP / 8, cudaMemcpyHostToDevice);

	//checkHostMatrix<unsigned char>(FreeImage_GetBits(tmp), imagePitch, imageHeight, imageWidth, "");
	//checkDeviceMatrix<unsigned char>(dImageData, imagePitch, imageHeight, imageWidth, "", "");

	FreeImage_Unload(tmp);
	FreeImage_DeInitialise();
}
#pragma endregion

#pragma region STEP 2

//TASK: Create a texture based on the source image. The input images can have variable BPP (Byte Per Pixel), but finally any such image will be converted into the floating-point texture using
//		the colorToFloat kernel.

void createSrcTexure()
{
	//Floating Point Texture Data
	cudaMallocPitch((void**)&dLinearPitchTextureData, &texPitch, imageWidth * sizeof(float), imageHeight);

	//Converts custom image data to float and stores result in the float_pitch_linear_data
	switch (imageBPP)
	{
	case 8:  colorToFloat<8, 2> << <squareKs.dimGrid, squareKs.dimBlock >> > (dImageData, imageWidth, imageHeight, imagePitch, texPitch / sizeof(float), dLinearPitchTextureData); break;
	case 16: colorToFloat<16, 2> << <squareKs.dimGrid, squareKs.dimBlock >> > (dImageData, imageWidth, imageHeight, imagePitch, texPitch / sizeof(float), dLinearPitchTextureData); break;
	case 24: colorToFloat<24, 2> << <squareKs.dimGrid, squareKs.dimBlock >> > (dImageData, imageWidth, imageHeight, imagePitch, texPitch / sizeof(float), dLinearPitchTextureData); break;
	case 32: colorToFloat<32, 2> << <squareKs.dimGrid, squareKs.dimBlock >> > (dImageData, imageWidth, imageHeight, imagePitch, texPitch / sizeof(float), dLinearPitchTextureData); break;
	}

	//checkDeviceMatrix<float>(dLinearPitchTextureData, texPitch, imageHeight, imageWidth, "", "");

	//Texture settings
	texChannelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	texRef.normalized = false;
	texRef.filterMode = cudaFilterModePoint;
	texRef.addressMode[0] = cudaAddressModeClamp;
	texRef.addressMode[1] = cudaAddressModeClamp;

	cudaBindTexture2D(0, &texRef, dLinearPitchTextureData, &texChannelDesc, imageWidth, imageHeight, texPitch);
}
#pragma endregion

#pragma region STEP 3

//TASK:	Convert the input image into normal map. Use the binded texture (srcTexRef).

void createNormalMap()
{
	size_t pitch;
	//TODO: Allocate Pitch memory dstTexData to store output texture
	checkCudaErrors(cudaMallocPitch((void**)&dstTexData, &texPitch, imageWidth * 3, imageHeight));

	//TODO: Call the kernel that creates the normal map.
	floatHeighmapTextureToNormalmap<true> << <squareKs.dimGrid, squareKs.dimBlock>> >(imageWidth, imageHeight, texPitch, dstTexData);

	//check_data<uchar3>::checkDeviceMatrix(dstTexData, imageHeight, texPitch / sizeof(uchar3), true, "%hhu %hhu %hhu | ", "Result of Linear Pitch Text");
}

#pragma endregion

#pragma region STEP 4

//TASK: Save output image (normal map)

void saveTexImage(const char* imageFileName)
{
	FreeImage_Initialise();

	FIBITMAP *tmp = FreeImage_Allocate(imageWidth, imageHeight, 24);
	unsigned int tmpPitch = imagePitch = FreeImage_GetPitch(tmp);		// FREEIMAGE align row data ... You have to use pitch instead of width
	checkCudaErrors(cudaMemcpy2D(FreeImage_GetBits(tmp), FreeImage_GetPitch(tmp), dstTexData, texPitch, imageWidth * 3, imageHeight, cudaMemcpyDeviceToHost));
	//FreeImage_Save(FIF_PNG, tmp, imageFileName, 0);
	ImageManager::GenericWriter(tmp, imageFileName, FIF_PNG);
	FreeImage_Unload(tmp);
	FreeImage_DeInitialise();
}

#pragma endregion

void releaseMemory()
{
	cudaUnbindTexture(texRef);
	if (dImageData != 0)
		cudaFree(dImageData);
	if (dLinearPitchTextureData != 0)
		cudaFree(dLinearPitchTextureData);
	if (dArrayTextureData)
		cudaFreeArray(dArrayTextureData);
	if (dOutputData)
		cudaFree(dOutputData);
}

int main(int argc, char *argv[])
{
	initializeCUDA(deviceProp);

	//STEP 1
	loadSourceImage("C:/Users/kne0035/dev/projects/parallelAlgorithm2/parallelAlgorithm2/images/terrain3Kx3K.tif");

	//TODO: Setup the kernel settings
	squareKs.dimBlock = dim3(BLOCK_DIM, BLOCK_DIM, 1);
	squareKs.blockSize = BLOCK_DIM * BLOCK_DIM;
	squareKs.dimGrid = dim3((imageWidth + BLOCK_DIM - 1) / BLOCK_DIM, (imageHeight + BLOCK_DIM - 1) / BLOCK_DIM, 1);

	//Step 2 - create heighmap texture stored in the linear pitch memory
	createSrcTexure();

	//Step 3 - create the normal map
	createNormalMap();

	//Step 4 - save the normal map
	saveTexImage("C:/Users/kne0035/dev/projects/parallelAlgorithm2/parallelAlgorithm2/images/normalMap.bmp");

	releaseMemory();
}
