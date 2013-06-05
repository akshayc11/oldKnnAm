//*******************************************************
// Author: Akshay Chandrashekaran
// Date: 10/16/2012
// Description: Header for load test data to CPU
//*******************************************************


#ifndef __ALIGNTEST_NEWPFILE_GPU_H__
#define __ALIGNTEST_NEWPFILE_GPU_H__

#include "../ITest.h"
#include "../../runopt.h"
#include "../../Train_factory/ITrain.h"

#include <string>
#include <fstream>

class AlignTest_newpFile_GPU : public ITest {
 public:
  AlignTest_newpFile_GPU(const Runopt* _options, ITrain* _train);
  ~AlignTest_newpFile_GPU();
  
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
  std::ifstream inFile;
  size_t headerSize;
  FILE *testFile;
  FILE *testLabel;
  float **hostObject;
  labelType* hostObjLabel;
  unsigned int offset;
};

#endif // __TEST_GPU_H__
