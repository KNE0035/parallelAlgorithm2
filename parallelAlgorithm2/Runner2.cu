#include <cudaDefs.h>


cudaDeviceProp deviceProp = cudaDeviceProp();

void createMatrixOnDevice(unsigned int mRows, unsigned int mCols);


int main(int argc, char *argv[])
{
	initializeCUDA(deviceProp);
	
	
	
	
	testAdding2Vectors();
	testAddingNVectors();
}


void createMatrixOnDevice(unsigned int mRows, unsigned int mCols) {
	unsigned int* dMatrix = new unsigned int[mRows * mCols];
	unsigned int* pitch;


	cudaMallocPitch((void**)&dMatrix, &pitch, )



}