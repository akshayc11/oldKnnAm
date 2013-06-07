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
#include <thrust/gather.h>

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
    exit(1);
  }
  
  try {
    trainDistances.resize(N);
  }
  catch (std::bad_alloc &e) {
    std::cerr << "Couldn't allocate trainDistances\n";
    exit(2);
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
  // std::cout << "getKNN()" << std::endl;
  try {
    trainLabel.resize(N);
  }
  catch (std::bad_alloc &e) {
    std::cerr << "Unable to allocate memory for trainLabel" <<std::endl;
  }
  thrust::host_vector <unsigned int> h_trainLabel (N);
  thrust::copy(trainObjLabel, 
	       trainObjLabel + N, 
	       h_trainLabel.begin());
  trainLabel = h_trainLabel;
  
  thrust::host_vector <float> host_tempDistances(K);
  thrust::host_vector <unsigned int> host_tempIndexes(K);
  
  std::cout << "K: " << K 
	    << " M: " << M 
	    << " D: " << D 
	    << " maxLabel: " << maxLabel 
	    << std::endl;
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
  
  thrust::host_vector <float> h_outVal;
  try {
    h_outVal.resize(K);
  }
  catch (thrust::system_error &e) {
    std::cerr << "Error in allocating memory for h_outVal:" 
	      << e.what() 
	      << std::endl;
    exit(3);
      
  }
  thrust::device_vector <float> outDistances;
  try {
    outDistances.resize(K);
  }
  catch (thrust::system_error &e) {
    std::cerr << "Error in allocating memory for outDistances:" 
	      << e.what() 
	      << std::endl;
    exit(4);
      
  }
  
  thrust::device_vector <float> outDistances2;
  try {
    outDistances2.resize(K);
  }
  catch (thrust::system_error &e) {
    std::cerr << "Error in allocating memory for outDistances2:" 
	      << e.what() 
	      << std::endl;
    exit(5);
      
  }
  thrust::device_vector <unsigned int> outIndexes;
  try {
    outIndexes.resize(K);
  }
  catch (thrust::system_error &e) {
    std::cerr << "Error in allocating memory for outIndexes:" 
	      << e.what() 
	      << std::endl;
    exit(6);
      
  }
  
  thrust::device_vector <unsigned int> outIndexes2;
  try {
    outIndexes2.resize(K);
  }
  catch (thrust::system_error &e) {
    std::cerr << "Error in allocating memory for outIndexes2:" 
	      << e.what() 
	      << std::endl;
    exit(7);
      
  }
  
  thrust::device_vector <unsigned int> outLabels;
  try {
    outLabels.resize(K);
  }
  catch (thrust::system_error &e) {
    std::cerr << "Error in allocating memory for outLabels:" 
	      << e.what() 
	      << std::endl;
    exit(6);
      
  }
  
  thrust::device_vector <unsigned int> outLabels2;
  try {
    outLabels2.resize(K);
  }
  catch (thrust::system_error &e) {
    std::cerr << "Error in allocating memory for outLabels2:" 
	      << e.what() 
	      << std::endl;
    exit(7);
      
  }
  
  
  thrust::device_vector<float> count;
  try {
    count.resize(K);
  }
  catch (thrust::system_error &e) {
    std::cerr << "Error in allocating memory for count" 
	      << e.what() 
	      << std::endl;
    exit(8);
      
  }
  thrust::device_vector<float> count2;
  try {
    count2.resize(K);
  }
  catch (thrust::system_error &e) {
    std::cerr << "Error in allocating memory for count" 
	      << e.what() 
	      << std::endl;
    exit(9);
      
  }
  
  float* objTest;
  
  if (cudaSuccess != cudaMalloc((void **)&objTest, D*sizeof(float))) {
    std::cerr << "Error in allocating global memory for objTest" 
	      << std::endl;
    exit(10);
  }
  
  
  unsigned int  maxTick = 1000;
  float*        trainDistance_ptr  = thrust::raw_pointer_cast(&trainDistances[0]);
  
  
  float*        distanceP          = (float* )        calloc (sizeof(float),       K*maxTick);
  unsigned int* indexP             = (unsigned int* ) calloc (sizeof(unsigned int),K*maxTick);
  
  unsigned int* label1P            = (unsigned int* ) calloc (sizeof(unsigned int),maxTick);
  unsigned int* label2P            = (unsigned int* ) calloc (sizeof(unsigned int),maxTick);
  
  unsigned int ticker              = 0;
  unsigned int globalTicker        = 0;
  
  std::cout << "creating temp arrays in GPU" << std::endl;
  
  thrust::device_vector <float> likelihood1;
  try {
     likelihood1.resize(maxLabel * maxTick);
  }
  catch (thrust::system_error &e) {
    std::cerr << "Error in allocating memory for likelihood1: " << e.what() << std::endl;
    exit(11);
      
  }
  thrust::device_vector <float> likelihood2;
  try {
    likelihood2.resize(maxLabel * maxTick);
  }
  catch (thrust::system_error &e) {
    std::cerr << "Error in allocating memory for likelihood2: " << e.what() << std::endl;
    exit(12);
      
  }
  thrust::device_vector <float> likelihood_p;
  try {
    likelihood_p.resize(maxLabel);
  }
  catch (thrust::system_error &e) {
    std::cerr << "Error in allocating memory for likelihood_p: " << e.what() << std::endl;
    exit(13);
      
  }
  
  std::cout << "Done" << std::endl;
  
  FILE*       fp      = fopen(opFileName.c_str(), "wb");
  
  std::string like1Op = opFileName + ".lik1";
  FILE*       fp1     = fopen(like1Op.c_str(), "wb");
  
  std::string like2Op = opFileName + ".lik2";
  FILE*       fp2     = fopen(like2Op.c_str(), "wb");

  boost::progress_display show_progress(M);

  for (unsigned int i= 0; i < M; i++) {
    //objTest = testObjectData + (i*D);
    // Copy data from device to host
    // std::cout <<"before copying test data point from host to device:" << i << std::endl;

    if (cudaSuccess != cudaMemcpy(objTest,
				  testObjectData + i*D,
				  D*sizeof(float),
				  cudaMemcpyHostToDevice)) {
      std::cerr << "Unable to copy point at index: " 
		<< i 
		<< " of test from host to device"
		<< std::endl;
      exit(14);
    }
    
    // std::cout <<"before computeDistance:" << i << std::endl;

    thrustKNN_GPU_computeDistance
      <<<  numBlocks, numThreadsPerBlock, blockSharedDataSize >>>
      (objTest,
       trainObjectData,
       N,
       D,
       trainDistance_ptr);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "Error: computeDistance kernel failure: " 
		<<  cudaGetErrorString(err) 
		<< std::endl;
      exit(15);
    }
    
    //cudaDeviceSynchronize();
    
      
    // std::cout <<"before computing KNN:" << i << std::endl;

    try {
      thrust::sequence(trainIndex.begin(), 
		       trainIndex.end(), 
		       0);
      //cudaDeviceSynchronize();
    }
    catch (thrust::system_error &e) {
      std::cerr << "Error in running sequence on trainIndex: "
		<< e.what() 
		<< std::endl;
      exit(16);
      
    }
    //cudaDeviceSynchronize();
    try {
      thrust::sort_by_key( trainDistances.begin(), 
			   trainDistances.end(),
			   trainIndex.begin());
      //cudaDeviceSynchronize();
    }
    catch (thrust::system_error &e) {
      std::cerr << "Error in sort_by_key for trainDistances and trainIndex: " 
		<< e.what() 
		<< std::endl;
      exit(18);
    }
    
    // At this point, the distances have been sorted. Now, transfer the 1 to Kth neighbor info to
    // a host vector
    // Assuming that the first nearest neighbor would be the point itself 
    //cudaDeviceSynchronize();
    // {
    //   // Temporary print of distances 
    //   thrust::copy_n(trainDistances.begin(), 
    // 		     K, 
    // 		     host_tempDistances.begin());
    //   cudaDeviceSynchronize();
    //   for (unsigned int k = 0; k < K; k++)
    // 	std::cout << host_tempDistances[k] << " ";
    //   std::cout << std::endl;
    // }
    // std::cout <<"before LL:" << i << std::endl;
    // cudaDeviceSynchronize();

    // Log Likelihood computation
    {
      // std::cout <<"before computing LL1:" << i << std::endl;

      try {
	thrust::copy_n(trainIndex.begin(), 
		       K, 
		       outIndexes.begin());
	//cudaDeviceSynchronize();

	thrust::gather(outIndexes.begin(), 
		       outIndexes.end(),
		       trainLabel.begin(),
		       outLabels.begin());
	
	thrust::copy(outLabels.begin(),
		     outLabels.end(),
		     outLabels2.begin());
	
	thrust::copy_n(trainDistances.begin(), 
		       K, 
		       outDistances.begin());
	//cudaDeviceSynchronize();

      }
      catch (thrust::system_error &e) {
	std::cerr << "Error in copying from trainIndex or trainDistances: " 
		  << e.what() 
		  << std::endl;
	exit(19);
	
      }
      //cudaDeviceSynchronize();
      // Method 1 of lik computation: Using only indexes:
      try {
	thrust::sort(outLabels.begin(), outLabels.end());
      }
      catch (thrust::system_error &e) {
	std::cerr << "Error in sorting outLabels:" 
		  << e.what() 
		  << std::endl;
	exit(20);
	
      }
      //cudaDeviceSynchronize();
	  
      try {
	thrust::fill(count.begin(), 
		     count.end(), 
		     1);
	//cudaDeviceSynchronize();

      }
      catch (thrust::system_error &e) {
	std::cerr << "Error in filling count:" 
		  << e.what() 
		  << std::endl;
	exit(21);
	
      }
      //cudaDeviceSynchronize();
      
      thrust::pair <thrust::device_vector<unsigned int>::iterator, 
    		    thrust::device_vector <float>::iterator > new_pair;
      try {
	new_pair = thrust::reduce_by_key(outLabels.begin(), 
					 outLabels.end(), 
					 count.begin(),
					 outLabels.begin(),
					 count.begin());
	//cudaDeviceSynchronize();

      }
      catch (thrust::system_error &e) {
	std::cerr << "Error in reducing count using outLabels:" 
		  << e.what() 
		  << std::endl;
	exit(22);
	
      }
      cudaDeviceSynchronize();
      unsigned int n = new_pair.second - count.begin();
      float fK = K;
      
      try {
	thrust::sort_by_key(count.begin(), 
			    count.begin() + n, 
			    outLabels.begin());
	//cudaDeviceSynchronize();

      }
      catch (thrust::system_error &e) {
	std::cerr << "Error in sorting outLabels using count:" 
		  << e.what() 
		  << std::endl;
	exit(22);
	
      }
      //cudaDeviceSynchronize();
      unsigned int label_p;
      try {
	label_p = outLabels[n-1];
	//cudaDeviceSynchronize();
      }
      catch (thrust::system_error &e) {
	std::cerr << "Error accessing vector element: outLabels["
		  << n-1
		  << "] : "
		  << e.what() 
		  << std::endl;
	exit(23);
	
      }
      //cudaDeviceSynchronize();
      label_p = BYTE_4_REVERSE(label_p);
      label1P[ticker] = label_p;
      
      
      try {
	thrust::transform(count.begin(), 
			  count.begin() + n, 
			  count.begin(), 
			  negLogDivideFunctor<float>(fK));
	//cudaDeviceSynchronize();
	
      }
      catch (thrust::system_error &e) {
	std::cerr << "Error in negLogDivideTransform of count:" 
		  << e.what() 
		  << std::endl;
	exit(24);
	
      }
      //cudaDeviceSynchronize();
      try {
	thrust::transform(count.begin(),
			  count.end(),
			  count.begin(),
			  flipEndian<float>());
	//cudaDeviceSynchronize();
      }
      catch (thrust::system_error &e) {
	std::cerr << "Error in flipEndian transform of count:" 
		  << e.what() 
		  << std::endl;
	exit(25);
	
      }
      //cudaDeviceSynchronize();
      
      try {
	thrust::fill(likelihood_p.begin(), 
		     likelihood_p.end(), 
		     score_max);
	//cudaDeviceSynchronize();
      }
      catch (thrust::system_error &e) {
	std::cerr << "Error in filling likelihood_p: :" 
		  << e.what() 
		  << std::endl;
	exit(26);
	
      }
      //cudaDeviceSynchronize();
      // {
      // 	std::cout << "n:" << n << std::endl;
      // 	for (unsigned int o = 0; o < outLabels.size(); o++)
      // 	  std::cout << outLabels[o] << " ";
      // }
      // std::cout << std::endl;
      try {
	thrust::scatter(count.begin(), 
		      count.begin() + n, 
		      outLabels.begin(), 
		      likelihood_p.begin());
	//cudaDeviceSynchronize();
      }
      catch (thrust::system_error &e) {
	std::cerr << "Error in scattering count based on outLabels:" 
		  << e.what() 
		  << std::endl;
	exit(27);
	
      }
      
      //cudaDeviceSynchronize();
      try {
	thrust::copy_n(likelihood_p.begin(),
		       maxLabel,
		       likelihood1.begin() + ticker*maxLabel);
	// std::cout << ticker*maxLabel << " " << likelihood1.size() << std::endl;
	//cudaDeviceSynchronize();
      }
      catch (thrust::system_error &e) {
	std::cerr << "Error in copy_n of likelihood_p to likelihood1 n: "
		  << maxLabel
		  << ": " 
		  << e.what() 
		  << std::endl;
	exit(28);
	
      }
      //cudaDeviceSynchronize();
      // std::cout <<"before computing LL2:" << i << std::endl;

      // method 2 of lik: using distances too:
      
      try {
	thrust::sort_by_key(outLabels2.begin(), 
			    outLabels2.end(), 
			    outDistances.begin());
	//cudaDeviceSynchronize();
      }
      catch (thrust::system_error &e) {
	std::cerr << "Error in sorting outDistances using outLabels2:" 
		  << e.what() 
		  << std::endl;
	exit(29);
	
      }
      //cudaDeviceSynchronize();
      // Area of problem: trying ot copy data from outLabels2 to outLabels.
      // std::cout << outLabels.size() << " " << outLabels2.size() << std::endl;
      try {
	thrust::copy(outLabels2.begin(), 
		     outLabels2.end(), 
		     outLabels.begin());
	//cudaDeviceSynchronize();
      }
      catch (thrust::system_error &e) {
	std::cerr << "Error in copying outLabels2 to outLabels: " 
		  << e.what() 
		  << std::endl;
	exit(30);
	
      }
      //cudaDeviceSynchronize();

      
      try {
	thrust::fill(count.begin(), 
		     count.end(), 
		     1);
	//cudaDeviceSynchronize();
      }
      catch (thrust::system_error &e) {
	std::cerr << "Error in filling count:" 
		  << e.what() 
		  << std::endl;
	exit(30);
	
      }

      //cudaDeviceSynchronize();

      try {
	thrust::reduce_by_key(outLabels.begin(),
			      outLabels.end(),
			      count.begin(),
			      outLabels.begin(),
			      count.begin());
	//cudaDeviceSynchronize();
      }
      catch (thrust::system_error &e) {
	std::cerr << "Error in reducing count using outLabels:" 
		  << e.what() 
		  << std::endl;
	exit(31);
	
      }
      //cudaDeviceSynchronize();
      try {
	thrust::transform(outDistances.begin(), 
			  outDistances.end(), 
			  outDistances.begin(), 
			  expXbySigma_functor<float>(Sigma2));
	//cudaDeviceSynchronize();
      }
      catch (thrust::system_error &e) {
	std::cerr << "Error in transformaing outDistances using expXbySigma:" 
		  << e.what() 
		  << std::endl;
	exit(32);
	
      }
      //cudaDeviceSynchronize();
      
      try { 
	thrust::reduce_by_key(outLabels2.begin(), 
			      outLabels2.end(), 
			      outDistances.begin(),
			      outLabels2.begin(),
			      outDistances.begin());
	//cudaDeviceSynchronize();
      }
      catch (thrust::system_error &e) {
	std::cerr << "Error in reduce_by key of outLabels2 and outDistances:" 
		  << e.what() 
		  << std::endl;
	exit(33);
	
      }
      //cudaDeviceSynchronize();
      
      try {
	thrust::transform(outDistances.begin(), 
			  outDistances.begin() + n, 
			  count.begin(), 
			  outDistances.begin(),
			  thrust::divides<float>());
	//cudaDeviceSynchronize();
      }
      catch (thrust::system_error &e) {
	std::cerr << "Error in dividing outDistances with count:" 
		  << e.what() 
		  << std::endl;
	exit(34);
	
      }
      //cudaDeviceSynchronize();
      try {
	thrust::sort_by_key(outDistances.begin(), 
			    outDistances.begin() + n, 
			    outLabels2.begin());
	//cudaDeviceSynchronize();
      }
      catch (thrust::system_error &e) {
	std::cerr << "Error in sorting outLabels2 using outDistance:" 
		  << e.what() 
		  << std::endl;
	exit(35);
      }
      
      label_p = outLabels[0];
      label_p = BYTE_4_REVERSE(label_p);
      label2P[ticker] = label_p;
      
      //cudaDeviceSynchronize();
      
      try {
	thrust::transform(outDistances.begin(), 
			  outDistances.begin() + n, 
			  outDistances.begin(), 
			  negLogFunctor<float>());
	//cudaDeviceSynchronize();
      }
      catch (thrust::system_error &e) {
	std::cerr << "Error in transforming outDistances with negLogFunctor:" 
		  << e.what() 
		  << std::endl;
	exit(36);
	
      }
      //cudaDeviceSynchronize();
      
      try {
	thrust::transform(outDistances.begin(),
			  outDistances.end(),
			  outDistances.begin(),
			  flipEndian<float>());
	//cudaDeviceSynchronize();
      }
      catch (thrust::system_error &e) {
	std::cerr << "Error in flipEndian for outDistances:" 
		  << e.what() 
		  << std::endl;
	exit(37);
	
      }
      //cudaDeviceSynchronize();
      
      try {
	thrust::fill(likelihood_p.begin(), 
		     likelihood_p.end(), 
		     score_max);
	//cudaDeviceSynchronize();
      }
      catch (thrust::system_error &e) {
	std::cerr << "Error in filling likelihood_p:" 
		  << e.what() 
		  << std::endl;
	exit(38);
      }
      //cudaDeviceSynchronize();
      
      try {
	thrust::scatter(outDistances.begin(), 
			outDistances.begin() + n, 
			outLabels2.begin(), 
			likelihood_p.begin());
	//cudaDeviceSynchronize();
      }
      catch (thrust::system_error &e) {
	std::cerr << "Error in scattering outDistances into likelihood_p using outLabels2:" 
		  << e.what() 
		  << std::endl;
	exit(39);
      }
      //cudaDeviceSynchronize();
      
      try {
	thrust::copy_n(likelihood_p.begin(),
		       maxLabel,
		       likelihood2.begin() + ticker*maxLabel);
	//cudaDeviceSynchronize();
      }
      catch (thrust::system_error &e) {
	std::cerr << "Error in copying likelihhod_p to likelihood2:" 
		  << e.what() 
		  << std::endl;
	exit(40);
      }
      //cudaDeviceSynchronize();
      
    }
    // std::cout <<"before copying KNN data:" << i << std::endl;

    //thrust::copy_n(trainIndex.begin()+1, K, d_trainIndex.begin()+(K*i) );
    try {
      thrust::copy_n(trainIndex.begin(), 
		     K, 
		     indexP + (K*ticker));
      //cudaDeviceSynchronize();
    }
    catch (thrust::system_error &e) {
      std::cerr << "Error in coyping trainIndex to indexP:" 
		<< e.what() 
		<< std::endl;
      exit(41);
      
    }
      
    //cudaDeviceSynchronize();
    
    //thrust::copy_n(trainDistances.begin()+1, K, d_trainDistances.begin()+(K*i) );
    try {
      thrust::copy_n(trainDistances.begin(), 
		     K, 
		     distanceP + (K*ticker));
      //cudaDeviceSynchronize();
    }
    catch (thrust::system_error &e) {
      std::cerr << "Error in copying trainDistances to distanceP:" 
		<< e.what() 
		<< std::endl;
      exit(42);
      
    }
      
    cudaDeviceSynchronize();
    
    ++show_progress;
    
    //cudaDeviceSynchronize();
    ticker++;
    if ((ticker % maxTick) == 0) {
      // std::cout <<"before writing to file in ticker:" << i << std::endl;

      // Write the data accumalated in the distanceP array to write buffer:
      
      thrust::host_vector <float> h_likelihood1 = likelihood1;
      thrust::host_vector <float> h_likelihood2 = likelihood2;
      cudaDeviceSynchronize();
      
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
  // std::cout <<"before writing remainder data:" << std::endl;

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
 
  
  // std::cout <<  std::endl;
  //thrust::host_vector <float> h_trainDistances    = d_trainDistances;
  //thrust::host_vector <unsigned int> h_trainIndex = d_trainIndex;

  // std::cout << "Done with computations. Now storing the results in output file" << std::endl;
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

