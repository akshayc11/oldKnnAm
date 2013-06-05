//**********************************************************************
// kNN: Test factory
// Author: Akshay Chandrashekaran
// Date: 08/16/2012
// Description: Factory Pattern for Test 
//**********************************************************************

#include <assert.h>
#include "../runopt.h"
#include "../Train_factory/ITrain.h"

#include "Test_implementation.h"
#include "Test_factory.h"

// #include "Test_CPU/Test_CPU.h"
// #include "Test_GPU/Test_GPU.h"

// #include "AlignTest_CPU/AlignTest_CPU.h"
#include "AlignTest_GPU/AlignTest_GPU.h"
#include "AlignTest_newpFile_GPU/AlignTest_newpFile_GPU.h"

#include <iostream>

using namespace std;

ITest* Test_factory::Get_Test(TestImplementation modelImpl, const Runopt* options, ITrain* train) {
  
  switch (modelImpl) {
  // case (TEST_CPU): return new Test_CPU(options);
  // case (TEST_GPU): return new Test_GPU(options);
  // case (ALIGNTEST_CPU): return new AlignTest_CPU(options, train);
  case (ALIGNTEST_GPU): return new AlignTest_GPU(options, train);
  case (ALIGNTEST_NEWPFILE_GPU): return new AlignTest_newpFile_GPU(options, train);
  default: cerr << "Error: Test factory cant find the specified model implementation" << endl;
  }
  return 0;
}
