/*
AlignKNN_GPU.h
Author: Akshay Chandrashekaran
Date: 10/22/2012
*/


#ifndef __ALIGNPARAKNN_GPU_H__
#define __ALIGNPARAKNN_GPU_H__

#include "../../runopt.h"
#include "../IParaKNN.h"

#include "../../shared/common.h"

class AlignParaKNN_GPU : public IParaKNN {
 public:
  std::string GetName();
  AlignParaKNN_GPU(const Runopt* options);
  ~AlignParaKNN_GPU();
  
  void CustomizeFunc();
  void PreExecution();
  void RunBatch();
  // Public in Parent Class:
  //ITrain* Train;
  //Protected in Parent Class:
  //const Runopt* options;

};


#endif //__ALIGNPARAKNN_GPU_H__
