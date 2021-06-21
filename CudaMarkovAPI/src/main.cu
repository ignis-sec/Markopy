#pragma once
#include <iostream>
#include <string>
#include <cstring>
#include <sstream>
#include <chrono>
#include "cudaModelMatrix.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>

using Markov::API::CUDA::CUDADeviceController;

int main(int argc, char** argv) {



	Markov::API::CUDA::CUDAModelMatrix markovPass;
	std::cerr << "Importing model.\n";
	markovPass.Import("models/finished.mdl");
	markovPass.Buff("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~", 10, true, true);
	std::cerr << "Import done. \n";
	markovPass.ConstructMatrix();
    //markovPass.DumpJSON();
	CUDADeviceController::ListCudaDevices();

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	std::cerr << "Starting walk. \n";
	markovPass.FastRandomWalk(1000000000,"/media/ignis/Stuff/wordlist.txt",12,12, false, false);
	//markovPass.FastRandomWalk(500000000,"/media/ignis/Stuff/wordlist2.txt",6,12,25, true);
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	std::cerr << "Finished in:" << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << " milliseconds" << std::endl;
	

}

