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
#ifndef __COMMON_CUDA_H__
#define __COMMON_CUDA_H__

#include <cutil.h>
#include "config.h"

#define myMAX(a,b)  ((a)>(b))?(a):(b)
#define myMIN(a,b)  ((a)<(b))?(a):(b)

inline int intDivideRoundUp(int a, int b){ return (a % b != 0) ? (a / b + 1) : (a / b);} 

#define FLOAT_TOLERANCE 0.005

__global__
void gpu_commonSelUniq(const int size,
                       //outputs
                       int *outArray,        // int*
                       //inputs
                       const int *inArray,   // int*
                       const int *ScanRes);  // int*


#endif
