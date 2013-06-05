//**********************************************************************
// kNN: Test factory
// Author: Akshay Chandrashekaran
// Date: 08/16/2012
// Description: Interface for Test 
//**********************************************************************

#ifndef __INTERFACE_TEST_H__
#define __INTERFACE_TEST_H__

#include <string>
#include "../runopt.h"
#include "../Train_factory/ITrain.h"

class ITest {
 public:
  
  virtual std::string GetName() = 0;
  virtual void Setup_Test()    = 0;
  virtual void Load_Test()     = 0;
  virtual void Clear_Test()    = 0;
  
  const Runopt* options;
  ITrain* train;
  
  unsigned int numCoords;
  unsigned int numObjs;
  int K;
  int Align;
  int Validation;
  float Split;
  float *objectData;
  labelType *objLabel;
  unsigned int* sentenceID;
  unsigned int* frameID;

};

#endif //__INTERFACE_TEST_H__
