/***********************************************************************
ParaKNN; thrustKNN_GPU.h
Author:/Akshay Chandrashekaran
Date: 10/28/2012
Description: header for KNN implementation on CPU
***********************************************************************/

#ifndef __thrustKNN_GPU_H__
#define __thrustKNN_GPU_H__

#include "../../Train_factory/ITrain.h"
#include "../../Test_factory/ITest.h"
#include "../../runopt.h"
//#include <algorithm>
//#include <vector>
#include <cuda.h>


// Thrust Libraries
#include <thrust/device_vector.h>
//#include <thrust/sort.h>
#include <cstdlib>


// CUSP libraries for sparse matrix:
#include <cusp/coo_matrix.h>



#include "../IKNN.h"

#define NTHREADS_PER_BLOCK_THRUSTKNN_GPU 512

class thrustKNN_GPU : public IKNN {
 public:
  thrustKNN_GPU(const Runopt* _options, ITrain* _Train, ITest* _Test);
  ~thrustKNN_GPU();
  
  std::string GetName();
  
  void Setup_KNN();
  void getKNN();
  
  labelType findKNN(unsigned int testIndex);
  
  void AllocateDataStructs();

  float computeDistance(float*       obj1,
			float*       obj2,
			unsigned int dim);
 private:
  
  unsigned int numBlocks;
  unsigned int numThreadsPerBlock;
  unsigned int blockSharedDataSize;
  std::string opFileName;
  float Sigma2;
  unsigned int* sentenceID;
  unsigned int* frameID;
  /* std::vector <kNodes> h_trainNodes; */
  /* std::vector <kNodes> h_testNodes; */
  thrust::device_vector <labelType> trainLabel;
  thrust::device_vector <unsigned int> trainIndex;
  thrust::device_vector <float> trainDistances;
  //float* trainDistances;
  thrust::device_vector <labelType> testLabel;
  thrust::device_vector <float> testDistances;
  thrust::device_vector <unsigned int> testIndex;
  thrust::device_vector <float> testDistrib;
  
  thrust::device_vector <unsigned int> tempTrainIndex;
  thrust::device_vector <labelType> tempTestLabels;
  thrust::device_vector <unsigned int> tempLabelCount;
  thrust::device_vector <float> tempVal;
  thrust::device_vector <float> tempTrainDistances;
  
  thrust::device_vector <labelType> outLabel;
  thrust::device_vector <float> outVal;
  thrust::device_vector <float> outDistances;
  thrust::device_vector <float> den;
  
  
  float* trainObjectData;
  float* testObjectData;
  labelType* trainObjLabel;
  labelType* testObjLabel;
  unsigned int D; // num of coords
  unsigned int N; // num objs in train
  unsigned int M; // num objs in test
  int          K;
};

#endif// __thrustKNN_GPU_H__
