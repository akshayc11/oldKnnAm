/***********************************************************************
ParaKNN: MP_GPU.h
Author:Akshay Chandrashekaran
Date: 03/05/2013
Description: header for MP implementation on CPU
***********************************************************************/

#ifndef __MP_GPU_H__
#define __MP_GPU_H__

#include "../../KNN_factory/IKNN.h"
#include "../../runopt.h"

#include <cuda.h>

#include <thrust/device_vector.h>
#include <cstdlib>

#include <cusp/coo_matrix.h>

#include "../IMP.h"

#define NTHREADS_PER_BLOCK_THRUSTKNN_GPU 512

class MP_GPU : public IMP {
 public:
  MP_GPU(const Runopt* options, IKNN* KNN);
  ~MP_GPU();
  std::string GetName();
  void Setup_MP();
  void AllocaterDataStructs();
  void getMP();
  
  int updateP();
  int updateQ();
  
};

#endif __MP_GPU_H__
