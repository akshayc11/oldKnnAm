/***********************************************************************
// IKNN.h
// Author: Akshay Chandrashekaran
// Date: 10/28/2012
// Description: KNN interface
***********************************************************************/

#ifndef __INTERFACE_KNN_H__
#define __INTERFACE_KNN_H__
#include <string>
#include <vector>
#include "../Train_factory/ITrain.h"
#include "../Test_factory/ITest.h"
#include "../runopt.h"
#include <thrust/device_vector.h>
#include <cusp/coo_matrix.h>
class IKNN {
 public:
  virtual std::string GetName() = 0;
  virtual void Setup_KNN() = 0;
  virtual void AllocateDataStructs() = 0;
  virtual void getKNN() = 0;

  const Runopt* options;
  
  ITrain* Train;
  ITest*  Test;

  
  labelType  maxLabel;
  int K;
  float Sigma;
  float V;
  float mu;
  float alpha;
  cusp::coo_matrix <unsigned int, float, cusp::device_memory> *sim_mat;
  thrust::device_vector <float> *gamma_vec;
  
};  
#endif //__INTERFACE_KNN_H__
