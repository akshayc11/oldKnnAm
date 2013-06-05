//  =============================================================
//  HYDRA - Common Functions for CUDA
//
//    common_cuda.h
//
//  -------------------------------------------------------------
//  CMU, 2011-
//
//  Description:
// 
//    Common Functions for CUDA
// 
//  ============================================================
#include "common_cuda.h"

__global__
void gpu_commonSelUniq(const int size,
                       //outputs
                       int *outArray,        // int*
                       //inputs
                       const int *inArray,   // int*
                       const int *scanRes){  // int*
  
  int tid = blockIdx.x*blockDim.x + threadIdx.x;

  if (tid<size) {
    if (inArray[tid] == 1){
      outArray[ scanRes[tid] ] = tid;
    }
  }
}


