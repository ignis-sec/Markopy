#pragma once
#include "MarkovModel/src/random.h"
#include "CudaMarkovPasswords/src/cudaDeviceController.h"

namespace Markov::API::CUDA::Random{

    class Marsaglia : public Markov::Random::Marsaglia, public CUDADeviceController{
	public:

		static unsigned long* MigrateToVRAM(Markov::API::CUDA::Random::Marsaglia *MEarr, long int gridSize){
			cudaError_t cudastatus;
			unsigned long* seedChunk;
			cudastatus = cudaMalloc((unsigned long**)&seedChunk, gridSize*3*sizeof(unsigned long));
			CudaCheckNotifyErr(cudastatus, "Failed to allocate seed buffer");
			unsigned long *temp = new unsigned long[gridSize*3];
			for(int i=0;i<gridSize;i++){
				temp[i*3]   = MEarr[i].x;
				temp[i*3+1] = MEarr[i].y;
				temp[i*3+2] = MEarr[i].z;
			}
			//for(int i=0;i<gridSize*3;i++) std::cout << temp[i] << "\n";
			cudaMemcpy(seedChunk, temp, gridSize*3*sizeof(unsigned long), cudaMemcpyHostToDevice);
			CudaCheckNotifyErr(cudastatus, "Failed to memcpy seed buffer.");
			return seedChunk;
		}
	};

	__device__ unsigned long devrandom(unsigned long &x, unsigned long &y, unsigned long &z){	
		unsigned long t;
		x ^= x << 16;
		x ^= x >> 5;
		x ^= x << 1;

		t = x;
		x = y;
		y = z;
		z = t ^ x ^ y;

		return z;
	}
};
