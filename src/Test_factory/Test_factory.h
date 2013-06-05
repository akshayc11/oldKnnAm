//**********************************************************************
// kNN: Test factory
// Author: Akshay Chandrashekaran
// Date: 08/16/2012
// Description: Functions for Test 
//**********************************************************************


#ifndef __TEST_FACTORY_H__
#define __TEST_FACTORY_H__

#include "ITest.h"
#include "Test_implementation.h"
#include "../runopt.h"
#include "../Train_factory/ITrain.h"

class Test_factory {

 public:
  static ITest* Get_Test(TestImplementation modelImpl, const Runopt* runopt);
  static ITest* Get_Test(TestImplementation modelImpl, const Runopt* runopt, ITrain* train);
};

#endif //__TEST_FACTORY_H__



