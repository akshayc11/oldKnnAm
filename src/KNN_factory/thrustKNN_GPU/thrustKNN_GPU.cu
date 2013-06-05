/***********************************************************************
ParaKNN: thrustKNN_GPU.cpp
Author:Akshay Chandrashekaran
Date: 10/28/2012
Description: KNN implementation on CPU
***********************************************************************/
#include <cuda.h>

#include <string>
#include <assert.h>
#include "thrustKNN_GPU.h"
#include "thrustKNN_GPU_kernel.h"

#include "../../Train_factory/ITrain.h"
#include "../../Test_factory/ITest.h"
#include <float.h>
#include <algorithm>
#include <vector>
#include <stdio.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/scatter.h>
// CUSP libraries for sparse matrix:
#include <cusp/coo_matrix.h>
#include <cusp/transpose.h>
#include <cusp/print.h>

#include <cusp/io/matrix_market.h>
#include <cusp/elementwise.h>

#include <cstdlib>
#include <cmath>


#include <iostream>
#include <boost/progress.hpp>

#define BYTE_4_REVERSE(x)       ((((x)<<24)&0xFF000000U) | (((x)<<8)&0x00FF0000U) | \
				 (((x)>>8) & 0x0000FF00U) | (((x)>>24)&0x000000FFU))

#define SCORE_MAX 999999.0

template <typename T>
struct saxpbFunctor {
  const T a;
  const T b;
  saxpbFunctor(T _a, T _b) : a( _a), b(_b) {}
  __host__ __device__ 
  T operator()(const T& x) const {
    return a*x + b;
  }
};
    
template <typename T>
struct sqrtFunctor {
  __host__ __device__
  T operator()(const T& p) const {
    return sqrt(p);
  }
};

template <typename T>
struct negLogFunctor {
  __host__ __device__
  T operator()(const T& p) const {
    return (-log(p));
  }
};


template <typename T>
struct flipEndian {
  __host__ __device__
  T operator()(const T& p) const {
    float tP = p;
    unsigned char *cptr = reinterpret_cast<unsigned char *> (&tP);
    unsigned char tmp = cptr[0];
    cptr[0] = cptr[3];
    cptr[3] = tmp;
    tmp = cptr[1];
    cptr[1] = cptr[2];
    cptr[2] = tmp;
    return (p);
  }
};

template <typename T>
struct negLogDivideFunctor {
  const float den;
  negLogDivideFunctor(float _den) : den(_den) {}
  __host__ __device__
  T operator()(const T& p) const {
    return (-log(p/den));
  }
};

//Sigma2 = 2*sigma*sigma
template <typename T>
struct expXbySigma_functor {
  const float Sigma2;
  
  expXbySigma_functor(float _Sigma2) : Sigma2(_Sigma2) {}
  __host__ __device__ 
  T operator () (const T& x) const {
    return (exp(-x/Sigma2)/2);
  }
};


std::string thrustKNN_GPU::GetName() {
  return "thrustKNN_GPU";
}

thrustKNN_GPU::thrustKNN_GPU(const Runopt* _options, ITrain* _Train, ITest* _Test) {
  options = _options;
  Train   = _Train;
  Test    = _Test;
  K = options->K;
  Sigma2 = (options->Sigma)*(options->Sigma);
  opFileName  = options->opFileName;
  
}

thrustKNN_GPU::~thrustKNN_GPU() {
  options = NULL;
  Train = NULL;
  Test = NULL;
  // if (h_trainNodes) delete(h_trainNodes);
  // if (h_testNodes ) delete(h_testNodes);
  
}
void thrustKNN_GPU::AllocateDataStructs() {
  

  N = Train->numObjs;
  M = Test->numObjs;
  D = Train->numCoords;
  maxLabel = Train->maxLabel;
  
  numThreadsPerBlock  = NTHREADS_PER_BLOCK_THRUSTKNN_GPU;
  numBlocks           = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;
  blockSharedDataSize = D*sizeof(float);
  
  trainObjectData  = Train->objectData;
  testObjectData   = Test->objectData;
  trainObjLabel    = Train->objLabel;
  testObjLabel     = Test->objLabel;
  sentenceID       = Test->sentenceID;
  frameID          = Test->frameID;
  //trainDistances = (float*) malloc(sizeof(float)*N);
  
  printf("thrustKNN_GPU:: AllocateDataStruct\n");
  
  try {
    trainIndex.resize(N);
  }
  catch (std::bad_alloc &e) {
    std::cerr << "Couldn't allocate trainIndex\n";
    exit(-1);
  }
  
  try {
    trainDistances.resize(N);
  }
  catch (std::bad_alloc &e) {
    std::cerr << "Couldn't allocate trainDistances\n";
    exit(-1);
  }
  
}

void thrustKNN_GPU::Setup_KNN() {
  //printf("thrustKNN_GPU::Setup_KNN\n");
}

void thrustKNN_GPU::getKNN() {
  
  // This function is to perform the kNN computations sequentially
  // on all the data in test using the data samples from train.
  // Functions to be implemented include: distance Metric. 

  // remember to free the mallocs here
  std::cout << "K: " << K << " M: " << M << " D: " << D << " maxLabel: " << maxLabel << std::endl;
  float score_max = SCORE_MAX;
  {  
    unsigned char *cptr = reinterpret_cast<unsigned char *>(&score_max);
    unsigned char tmp = cptr[0];
    cptr[0] = cptr[3];
    cptr[3] = tmp;
    tmp = cptr[1];
    cptr[1] = cptr[2];
    cptr[2] = tmp;      
  }
  printf("thrustKNN_GPU::findKNN_Batch\n");
  boost::progress_display show_progress(M);

  thrust::host_vector <float> h_outVal(K);
  
  
  thrust::device_vector <float> outDistances(K);
  thrust::device_vector <float> outDistances2(K);

  
  thrust::device_vector <unsigned int> outIndexes(K);
  thrust::device_vector <unsigned int> outIndexes2(K);

  
  thrust::device_vector<float> count(K);
  thrust::device_vector<float> count2(K);
  
  
  float* objTest;
  cudaMalloc((void **)&objTest, D*sizeof(float));
  
  
  unsigned int maxTick = 1000;
  float* trainDistance_ptr = thrust::raw_pointer_cast(&trainDistances[0]);
  
  
  float* distanceP         = (float* )        calloc (sizeof(float),D*maxTick);
  unsigned int* indexP     = (unsigned int* ) calloc (sizeof(unsigned int),D*maxTick);
  
  unsigned int* label1P     = (unsigned int* ) calloc (sizeof(unsigned int),maxTick);
  unsigned int* label2P     = (unsigned int* ) calloc (sizeof(unsigned int),maxTick);
  
  unsigned int ticker       = 0;
  unsigned int globalTicker = 0;
  
  thrust::device_vector <float> likelihood1 (maxLabel*maxTick); 
  thrust::device_vector <float> likelihood2 (maxLabel*maxTick);
  
  thrust::device_vector <float> likelihood_p(maxLabel);
  
  
  FILE* fp = fopen(opFileName.c_str(), "wb");
  
  std::string like1Op = opFileName + ".lik1";
  FILE* fp1 = fopen(like1Op.c_str(), "wb");
  
  std::string like2Op = opFileName + ".lik2";
  FILE* fp2 = fopen(like2Op.c_str(), "wb");
  
  for (unsigned int i= 0; i < M; i++) {
    //objTest = testObjectData + (i*D);
    // Copy data from device to host
    cudaMemcpy(objTest,
	       testObjectData + i*D,
	       D*sizeof(float),
	       cudaMemcpyHostToDevice);
    
    thrustKNN_GPU_computeDistance
      <<<  numBlocks, numThreadsPerBlock, blockSharedDataSize >>>
      (objTest,
       trainObjectData,
       N,
       D,
       trainDistance_ptr);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
      printf("Error: %s\n", cudaGetErrorString(err));
    
    cudaDeviceSynchronize();

    thrust::sequence(trainIndex.begin(), trainIndex.end(), 0);
    cudaDeviceSynchronize();
    try {
      thrust::sort_by_key( trainDistances.begin(), trainDistances.end(),trainIndex.begin());
    }
    catch (thrust::system_error &e) {
      std::cerr <<"Error in sort_by_key " << e.what() << std::endl;
    }
    // At this point, the distances have been sorted. Now, transfer the 1 to Kth neighbor info to
    // a host vector
    // Assuming that the first nearest neighbor would be the point itself 
    cudaDeviceSynchronize();

    // Log Likelihood computation
    {
      thrust::copy_n(trainIndex.begin(), K, outIndexes.begin());
      thrust::copy_n(trainIndex.begin(), K, outIndexes2.begin());
      thrust::copy_n(trainDistances.begin(), K, outDistances.begin());
      
      // Method 1 of lik computation: Using only indexes:
      thrust::sort(outIndexes.begin(), outIndexes.end());
      thrust::fill(count.begin(), count.end(), 1);
      thrust::pair <thrust::device_vector<unsigned int>::iterator, 
    		    thrust::device_vector <float>::iterator > new_pair;
      new_pair = thrust::reduce_by_key(outIndexes.begin(), 
    				       outIndexes.end(), 
    				       count.begin(),
    				       outIndexes.begin(),
    				       count.begin());
      
      unsigned int n = new_pair.second - count.begin();
      float fK = K;
      
      
      thrust::sort_by_key(count.begin(), count.begin() + n, outIndexes.begin());
      
      unsigned int label_p = outIndexes[n-1];
      label_p = BYTE_4_REVERSE(label_p);
      label1P[ticker] = label_p;
      
      
      
      thrust::transform(count.begin(), 
    			count.begin() + n, 
    			count.begin(), 
    			negLogDivideFunctor<float>(fK));
    
      
      thrust::transform(count.begin(),
			count.end(),
			count.begin(),
			flipEndian<float>());
      
      
      thrust::fill(likelihood_p.begin(), 
		   likelihood_p.end(), 
		   score_max);
      thrust::scatter(count.begin(), 
		      count.begin() + n, 
		      outIndexes.begin(), 
		      likelihood_p.begin());
      
      thrust::copy_n(likelihood_p.begin(),maxLabel,likelihood1.begin() + ticker*maxLabel);

      
      // method 2 of lik: using distances too:
      
      thrust::sort_by_key(outIndexes2.begin(), outIndexes2.end(), outDistances.begin());
      thrust::transform(outDistances.begin(), 
    			outDistances.end(), 
    			outDistances.begin(), 
    			expXbySigma_functor<float>(Sigma2));
    
      thrust::reduce_by_key(outIndexes2.begin(), 
    			    outIndexes2.end(), 
    			    outDistances.begin(),
    			    outIndexes2.begin(),
    			    outDistances.begin());
    
      thrust::transform(outDistances.begin(), 
    			outDistances.begin() + n, 
    			count.begin(), 
    			outDistances.begin(),
    			thrust::divides<float>());
      
      thrust::sort_by_key(outDistances.begin(), outDistances.begin() + n, outIndexes2.begin());
      label_p = outIndexes[0];
      label_p = BYTE_4_REVERSE(label_p);
      label1P[ticker] = label_p;
      
      
      thrust::transform(outDistances.begin(), 
    			outDistances.begin() + n, 
    			outDistances.begin(), 
    			negLogFunctor<float>());
      
      thrust::transform(outDistances.begin(),
			outDistances.end(),
			outDistances.begin(),
			flipEndian<float>());
      
      thrust::fill(likelihood_p.begin(), 
		   likelihood_p.end(), 
		   score_max);
      
      thrust::scatter(outDistances.begin(), 
		      outDistances.begin() + n, 
		      outIndexes2.begin(), 
		      likelihood_p.begin());

      thrust::copy_n(likelihood_p.begin(),maxLabel,likelihood2.begin() + ticker*maxLabel);
      
    }
    //thrust::copy_n(trainIndex.begin()+1, K, d_trainIndex.begin()+(K*i) );
    thrust::copy_n(trainIndex.begin(), K, indexP + (K*ticker));
    
    cudaDeviceSynchronize();
    
    //thrust::copy_n(trainDistances.begin()+1, K, d_trainDistances.begin()+(K*i) );
    thrust::copy_n(trainDistances.begin(), K, distanceP + (K*ticker));
    
    cudaDeviceSynchronize();
    
    ++show_progress;
    
    //cudaDeviceSynchronize();
    ticker++;
    
    if ((ticker % maxTick) == 0) {
      // Write the data accumalated in the distanceP array to write buffer:
      thrust::host_vector <float> h_likelihood1 = likelihood1;
      thrust::host_vector <float> h_likelihood2 = likelihood2;
      for (unsigned int i_tick = 0; i_tick < maxTick; i_tick++) {
	unsigned int* inP   = indexP    + (i_tick * K);
	float*        distP = distanceP + (i_tick * K);
	unsigned int ind = (globalTicker + i_tick);
	//unsigned int flip_ind = BYTE_4_REVERSE(ind);
	unsigned int sid = sentenceID[ind];
	unsigned int fid = frameID[ind];
	{
	  fwrite(&ind,  sizeof(unsigned int), 1, fp);
	  fwrite(inP,   sizeof(unsigned int), K, fp);
	  fwrite(distP, sizeof(float),        K, fp);
	}
	{
	  fwrite(&sid, sizeof(unsigned int), 1,fp1);
	  fwrite(&fid, sizeof(unsigned int), 1,fp1);
	  fwrite(&(h_likelihood1[i_tick*maxLabel]), sizeof(float), maxLabel,fp1);
	  fwrite(&(label1P[i_tick]), sizeof(unsigned int),1,fp1);
	}
	
	{
	  fwrite(&sid, sizeof(unsigned int), 1,fp2);
	  fwrite(&fid, sizeof(unsigned int), 1,fp2);
	  fwrite(&(h_likelihood2[i_tick*maxLabel]), sizeof(float), maxLabel,fp2);
	  fwrite(&(label2P[i_tick]), sizeof(unsigned int),1,fp2);
	}
      }
      globalTicker = globalTicker + ticker;
      ticker = 0;
    }
      
  }
  // Write the remaining data accumalated in the distanceP array to write buffer:
  for (unsigned int i_tick = 0; i_tick < ticker; i_tick++) {
    unsigned int* inP   = indexP    + (i_tick * K);
    float*        distP = distanceP + (i_tick * K);
    unsigned int ind = (globalTicker + i_tick);
    fwrite(&ind,  sizeof(unsigned int), 1, fp);
    fwrite(inP,   sizeof(unsigned int), K, fp);
    fwrite(distP, sizeof(float),        K, fp);
  }
  globalTicker = globalTicker + ticker;
  ticker = 0;
 
  
  std::cout <<  std::endl;
  //thrust::host_vector <float> h_trainDistances    = d_trainDistances;
  //thrust::host_vector <unsigned int> h_trainIndex = d_trainIndex;

  std::cout << "Done with computations. Now storing the results in output file" << std::endl;
  // At this point, all points have been used to find the K nearest neighbors 
  // Now, to write it out into a file
  if (fp == NULL) {
    std::cerr << "Unable to open output file" << std::endl;
    exit(1);
  }
  //int Size = h_trainIndex.size();
  //assert(Size == K*M);
  // for (unsigned int i = 0; i < M; i++) {
  //   fwrite(&i,sizeof(unsigned int),1,fp);
  //   fwrite(&(h_trainIndex[(i*K)]),     sizeof(unsigned int), K, fp);
  //   fwrite(&(h_trainDistances[(i*K)]), sizeof(float),        K, fp);
  // }
  cudaFree(objTest);
  free(distanceP);
  free(indexP);
  free(label1P);
  free(label2P);
  fclose(fp);    
  fclose(fp1);
  fclose(fp2);
}
float thrustKNN_GPU::computeDistance(float*       obj1, 
				     float*       obj2, 
				     unsigned int numDim) {
  float sum = 0;
  for (unsigned int i = 0; i< numDim; i++) {
    float X = obj1[i];
    float Y = obj2[i];
    float Z = X-Y;
    sum += (Z*Z);
  }
  return sum;
}

