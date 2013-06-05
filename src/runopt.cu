//======================================================================
// featAcces - Command Line Interface Parsing
// 
// runopt.cu
//----------------------------------------------------------------------
//  Copyright (c) Carnegie Mellon University, 2012
//
// Description:
//
//   GPU related mothods for command line options
//
// Revisions:
//   v0.1: 2012/06/16
//        Akshay Chandrashekaran (akshay.chandrasekaran@sv.cmu.edu)
//        Initial Implementation
//
//=====================================================================

#include <stdio.h>
#include "config.h"
#include "runopt.h"

void Runopt::detectGPU() {
  
  //Discover GPU devices
  //
  int deviceCount;
  cudaDeviceProp deviceProp;
  cudaGetDeviceCount(&deviceCount);
  
  for ( int dev = 0; dev<deviceCount; ++dev) {
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("\nDevice %d: \"%s\" at clock rate: %0.2f Ghz\n",
	   dev, deviceProp.name,
	   deviceProp.clockRate* 1e-6f);
    
    // If the GPU meets the minimum requirements, add it to vector
    // Devite 1.3 1 GB memory
    if (((deviceProp.major*10 + deviceProp.minor) >= (GPU_MAJOR*10 + GPU_MINOR)) &&
	(deviceProp.totalGlobalMem >= GPU_MEMORY)) {
            gpuIds.push_back(dev);
      gpuType.push_back((deviceProp.major*10 + deviceProp.minor));
      gpuMem.push_back(deviceProp.totalGlobalMem);
    }
  }

  if (FORCE_GPU_ID != -1){
    if (FORCE_GPU_ID >= gpuIds.size()){
      printf("ERROR: GPU %d not available.\n", FORCE_GPU_ID);
      exit(0);
    }

    cudaSetDevice(FORCE_GPU_ID);

    cudaGetDeviceProperties(&deviceProp, FORCE_GPU_ID);
    if(!QUIET) printf("INFO: Forced to use device %d, %s at %.2f GHz, device memory %.3f GB\n",
                      FORCE_GPU_ID,
                      deviceProp.name,
                      deviceProp.clockRate * 1e-6f,
                      deviceProp.totalGlobalMem * pow(2.0, -30.0));
    currDeviceMemSize = deviceProp.totalGlobalMem * pow(2.0, -20.0);
  } else {
    cudaSetDevice(0);
    cudaGetDeviceProperties(&deviceProp, 0);
    if(!QUIET) printf("Processing On Device: 0, %s at %.2f GHz, device memory %.3f GB\n",
                      deviceProp.name,
                      deviceProp.clockRate * 1e-6f,
                      deviceProp.totalGlobalMem  * pow(2.0, -30.0));
    currDeviceMemSize = deviceProp.totalGlobalMem * pow(2.0, -20.0);
  }

  size_t avail, total;
	cudaMemGetInfo( &avail, &total );
	size_t used = total - avail;
	printf("INFO: Device memory available: %1.3f MB  used: %1.3f MB \n",
         avail * pow(2.0,-20.0), used * pow(2.0,-20.0));
  return;
}

