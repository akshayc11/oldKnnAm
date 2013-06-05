#include "thrustKNN_GPU_kernel.h"
#include <stdio.h>

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
				   float* trainDistances) {

  // Move observation to shared memory
  extern __shared__ char sharedMemory[];
  float* sharedObs = (float *) sharedMemory;
  unsigned int tx   = threadIdx.x;
  unsigned int bx   = blockIdx.x;
  unsigned int bdim = blockDim.x;
  
  
  // Load observation into shared memory
  if (bdim > D) {
    for (unsigned int i = tx; i < D; i+= bdim) {
      sharedObs[i] = observation[i];
    }
  }
  else {
    for (int i = 0; i < D;i++)
      sharedObs[i] = observation[i];
    
  }
  __syncthreads();

  unsigned int trainObjectId = tx + bx*bdim;
  
  if (trainObjectId < N) {
    trainDistances[trainObjectId] = euc_dist2_thrustKNN_GPU(D, 
							    N,
							    trainObjectData,
							    sharedObs,
							    trainObjectId);

    //printf("ans : %0.1f checkDist = %0.1f\n",ans, trainDistances[trainObjectId]);
  }
  __syncthreads();
}

__host__ __device__ 
float euc_dist2_thrustKNN_GPU(unsigned int D,
			      unsigned int N,
			      float* trainObjectData,
			      float* sharedObs,
			      unsigned int objectId) {
  float ans = 0;
  // for (unsigned int i = 0; i < D; i++) {
  //   if (i%28 == 0)
  //     printf("\n");
  //   printf("%0.1f\t",trainObjectData[i*N + objectId]);
    
  // }
  // printf("\n");
  // for (unsigned int i = 0; i < D; i++) {
  //   if (i%28 == 0)
  //     printf("\n");
  //   printf("%0.1f\t",sharedObs[i]);
    
  // }
  // printf("\n");
  
  for (unsigned int i = 0; i < D; i++) {
    float temp = sharedObs[i] - trainObjectData[i*N + objectId];
    ans+=(temp*temp);
  }
  // printf("Distance = %0.1f\n", ans);
  return (ans);
}
