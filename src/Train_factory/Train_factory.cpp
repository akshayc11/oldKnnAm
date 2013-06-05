//**********************************************************************
// kNN: Train factory
// Author: Akshay Chandrashekaran
// Date: 08/16/2012
// Description: Factory Pattern for Train 
//**********************************************************************

#include <assert.h>
#include "../runopt.h"
#include "Train_implementation.h"
#include "Train_factory.h"

// #include "Train_CPU/Train_CPU.h"
// #include "Train_GPU/Train_GPU.h"
// #include "AlignTrain_CPU/AlignTrain_CPU.h"
#include "AlignTrain_GPU/AlignTrain_GPU.h"
#include "AlignTrain_newpFile_GPU/AlignTrain_newpFile_GPU.h"

#include <iostream>

using namespace std;

ITrain* Train_factory::Get_Train(TrainImplementation modelImpl, const Runopt* options) {
  
  switch (modelImpl) {
  // case (TRAIN_CPU): return new Train_CPU(options);
  // case (TRAIN_GPU): return new Train_GPU(options);
  // case (ALIGNTRAIN_CPU): return new AlignTrain_CPU(options);
  case (ALIGNTRAIN_GPU): return new AlignTrain_GPU(options);
  case (ALIGNTRAIN_NEWPFILE_GPU): return new AlignTrain_newpFile_GPU(options);
  default: cerr << "Error: Train factory cant find the specified model implementation" << endl;
  }
  return 0;
}
