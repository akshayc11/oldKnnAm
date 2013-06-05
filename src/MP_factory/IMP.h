/***********************************************************************
// IMP.h
// Author: Akshay Chandrashekaran
// Date: 2/28/2013
// Description: Measure Propagation interface
***********************************************************************/

#ifndef __INTERFACE_MP_H__
#define __INTERFACE_MP_H__

#include <string>
#include <vector>
#include "../KNN_factory/IKNN.h"
#include "../runopt.h"

#include <thrust/device_vector.h>
#include <cusp/coo_matrix.h>


class IMP {
 private:
  int n_iters;
  IKNN* KNN;
  const Runopt* options;
  
 public:
  virtual std::string GetName() = 0;
  virtual void Setup_MP() = 0;
  virtual void AllocateDataStructs() = 0;
  virtual void getMP() = 0;
  
  thrust::device_vector <float> *sim_row;
  thrust::device_vector <float> *sim_col;
  cusp::coo_matrix <unsigned int, float, cusp::device_memory> *sim_mat;
  
};
#endif
