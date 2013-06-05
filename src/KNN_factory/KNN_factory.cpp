/***********************************************************************
kNN: KNN_factory.cpp
Author:/Akshay Chandrashekaran
Date: 10/28/2012
Description: Factory Pattern for KNN
**********************************************************************/

#include <assert.h>

#include "../runopt.h"
#include "KNN_implementation.h"
#include "KNN_factory.h"

// #include "KNN_CPU/KNN_CPU.h"
// #include "KNN_GPU/KNN_GPU.h"
// #include "thrustKNN_CPU/thrustKNN_CPU.h"
#include "thrustKNN_GPU/thrustKNN_GPU.h"

#include "../Train_factory/ITrain.h"
#include "../Test_factory/ITest.h"

#include <iostream>

using namespace std;

IKNN* KNN_factory::Get_KNN(KNNImplementation modelImpl, const Runopt* options, ITrain* Train, ITest* Test) {
  switch (modelImpl) {
  // case(kNN_CPU):       return new KNN_CPU(options, Train, Test);
  // case(kNN_GPU):       return new KNN_GPU(options, Train, Test);
  // case(THRUST_KNN_CPU): return new thrustKNN_CPU(options, Train, Test);
  case(THRUST_KNN_GPU): return new thrustKNN_GPU(options, Train, Test);
  default: cerr << "Error: KNN factory cant find the specified model implementation" << endl;
  }
  return 0;
}
