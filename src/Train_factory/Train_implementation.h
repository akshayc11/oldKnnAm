//**********************************************************************
// kNN: Train factory
// Author: Akshay Chandrashekaran
// Date: 08/16/2012
// Description: List of Implementations 
//**********************************************************************

#ifndef __TRAIN_IMPLEMENTATION_H__
#define __TRAIN_IMPLEMENTATION_H__

enum TrainImplementation {
  TRAIN_CPU,
  TRAIN_GPU,
  ALIGNTRAIN_CPU,
  ALIGNTRAIN_GPU,
  ALIGNTRAIN_NEWPFILE_GPU
};

#endif //__TRAIN_IMPLEMENTATION_H__
