//*******************************************************
// Author: Akshay Chandrashekaran
// Date: 10/16/2012
// Description: Header for load test data to CPU
//*******************************************************


#ifndef __ALIGNTEST_CPU_H__
#define __ALIGNTEST_CPU_H__

#include "../ITest.h"
#include "../../runopt.h"
#include "../../Train_factory/ITrain.h"

#include <string>
#include <fstream>

class AlignTest_CPU : public ITest {
 public:
  AlignTest_CPU(const Runopt* _options, ITrain* _train);
  ~AlignTest_CPU();
  
  std::string GetName();
  void Setup_Test();
  void Load_Test();
  void Clear_Test();
  
  // Variables declared in ITest.h
  // int numCoords;
  // int numObjs;
  // int K;
  // float ** object;
  // labelType *objLabel;
  
  // Other required Variables
 private:
  FILE *testFile;
  FILE *testLabel;
  int offset;
};

#endif // __ALIGNTEST_CPU_H__
