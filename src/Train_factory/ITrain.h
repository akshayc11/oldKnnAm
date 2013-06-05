//**********************************************************************
// kNN: Train factory
// Author: Akshay Chandrashekaran
// Date: 08/16/2012
// Description: Interface for Train 
//**********************************************************************

#ifndef __INTERFACE_TRAIN_H__
#define __INTERFACE_TRAIN_H__

#include <string>
#include "../runopt.h"

class ITrain {
 public:
  
  virtual std::string GetName() = 0;
  virtual void Setup_Train()    = 0;
  virtual void Load_Train()     = 0;
  virtual void Clear_Train()    = 0;
  
  const Runopt* options;
  
  unsigned int numCoords;
  unsigned int numObjs;
  int K;
  int Align;
  int Validation;
  float Split;
  labelType maxLabel;
  float *objectData;
  labelType *objLabel;
  unsigned int *sentenceID;
  unsigned int *frameID;
  // Related to splitting of data into test and train
  int validNumObjs;
  float* validTest;
  labelType *validLabel;
};

#endif //__INTERFACE_TRAIN_H__
