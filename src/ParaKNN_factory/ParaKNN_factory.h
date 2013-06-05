/* 
kNN factory implementation:
Author: Akshay Chandrashekaran
Date: 10/22/2012
Description: Factory for Parallel KNN implementation
*/

#ifndef __PARAKNN_FACTORY_H__
#define __PARAKNN_FACTORY_H__

#include "IParaKNN.h"
#include "ParaKNN_implementation.h"

class ParaKNN_factory {
 public:
  static IParaKNN *Get_ParaKNN(ParaKNNImplementation modelImpl, const Runopt* _options);
};
#endif  //__PARAKNN_FACTORY_H__
