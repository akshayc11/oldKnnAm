/***********************************************************************
kNN: KNN_factory.h
Author: Akshay Chandrashekaran
Date: 10/28/2012
Description: Functions in KNN

**********************************************************************/

#ifndef __KNN_FACTORY_H__
#define __KNN_FACTORY_H__

#include "IKNN.h"
#include "KNN_implementation.h"

#include "../runopt.h"
#include "../Train_factory/ITrain.h"
#include "../Test_factory/ITest.h"

class KNN_factory {
 public:
  static IKNN* Get_KNN(KNNImplementation modelImpl, const Runopt* options, ITrain* _train, ITest* _test);
};

#endif //__KNN_FACTORY_H__
