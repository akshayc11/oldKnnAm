#ifndef __THRUSTKNN_GPU_KERNEL_H__
#define __THRUSTKNN_GPU_KERNEL_H__

#include <cuda.h>


// This function is a CUDA kernel that implements the distance computation for
// the given observation and the training data.
// The observation is to be stored in shared memory since it is to be used repeatedly
// next, the function follows a procedure similar to the kmeans compute eucledian 
// distance.


__global__
void thrustKNN_GPU_computeDistance(// Inputs
				   float* observation,
				   float* trainObjectData,
				   int N, // Number of objects in training data
				   int D, // Number of dimensions
				   
				   // Output
				   float* trainDistances);
__host__ __device__ 
float euc_dist2_thrustKNN_GPU(unsigned int D,
			      unsigned int N,
			      float* trainObjectData,
			      float* sharedObs,
			      unsigned int objectId);


#endif // __THRUSTKNN_GPU_KERNEL_H__
