/*
AlignKNN_CPU.h
Author: Akshay Chandrashekaran
Date: 10/22/2012
*/


#ifndef __ALIGNPARAKNN_CPU_H__
#define __ALIGNPARAKNN_CPU_H__

#include "../../runopt.h"
#include "../IParaKNN.h"

#include "../../shared/auxModels.h"
#include "../../shared/common.h"

class AlignParaKNN_CPU : public IParaKNN {
 public:
  std::string GetName();
  AlignParaKNN_CPU(const Runopt* options);
  ~AlignParaKNN_CPU();
  
  void CustomizeFunc();
  void PreExecution();
  void RunBatch();
  // Public in Parent Class:
  //ITrain* Train;
  //Protected in Parent Class:
  //const Runopt* options;

};


#endif //__ALIGNPARAKNN_CPU_H__
