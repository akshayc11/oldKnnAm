//**********************************************************************
// kNN: Train factory
// Author: Akshay Chandrashekaran
// Date: 08/16/2012
// Description: Functions for Train 
//**********************************************************************


#ifndef __TRAIN_FACTORY_H__
#define __TRAIN_FACTORY_H__

#include "ITrain.h"
#include "Train_implementation.h"
#include "../runopt.h"


class Train_factory {

 public:
  static ITrain* Get_Train(TrainImplementation modelImpl, const Runopt* runopt);
};

#endif //__TRAIN_FACTORY_H__



