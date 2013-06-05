//*******************************************************
// Author: Akshay Chandrashekaran
// Date: 10/16/2012
// Description: Header for load test data to CPU
//*******************************************************


#ifndef __TEST_GPU_H__
#define __TEST_GPU_H__

#include "../ITest.h"
#include "../../runopt.h"
#include <string>
#include <fstream>

class Test_GPU : public ITest {
 public:
  Test_GPU(const Runopt* _options);
  ~Test_GPU();
  
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
  float **hostObject;
  labelType* hostObjLabel;
   
};

#endif // __TEST_GPU_H__
