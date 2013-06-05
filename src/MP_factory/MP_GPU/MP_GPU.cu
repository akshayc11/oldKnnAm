/***********************************************************************
ParaKNN: MP_GPU.cu
Author:Akshay Chandrashekaran
Date: 03/05/2013
Description: MP implementation on GPU version 1
***********************************************************************/

#include "MP_GPU.h"

#include <cusp/coo_matrix.h>
#include <cusp/device_vector.h>

std::String MP_GPU::GetName() {
  return "MP_GPU";
}

MP_GPU::MP_GPU(const Runopt* _options, IKNN* _KNN) {
  
  options = _options;
  KNN     = _KNN;
  
  n_iters = options->n_iters;
  
  
}


