/* 
kNN factory implementation:
Author: Akshay Chandrashekaran
Date: 10/22/2012
Description: Factory for Parallel KNN implementation
*/


#include <assert.h>
#include "../runopt.h"
#include "ParaKNN_implementation.h"
#include "ParaKNN_factory.h"

// #include "ParaKNN_CPU/ParaKNN_CPU.h"
// #include "ParaKNN_GPU/ParaKNN_GPU.h"
// #include "AlignParaKNN_CPU/AlignParaKNN_CPU.h"
#include "AlignParaKNN_GPU/AlignParaKNN_GPU.h"

#include <iostream>

using namespace std;

IParaKNN* ParaKNN_factory::Get_ParaKNN(ParaKNNImplementation modelImpl, const Runopt* _options) {
  switch(modelImpl) {
  // case PARAKNN_CPU: return new ParaKNN_CPU(_options);
  // case PARAKNN_GPU: return new ParaKNN_GPU(_options);
  // case ALIGNPARAKNN_CPU: return new AlignParaKNN_CPU(_options);
  case ALIGNPARAKNN_GPU: return new AlignParaKNN_GPU(_options);
  default:
    cerr << "Error: ParaKNN_factory: model implementation not found" << endl;
  }
  return 0;
}
