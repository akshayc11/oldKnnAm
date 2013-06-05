/*
AlignParaKNN_GPU.cpp
Author: Akshay Chandrashekaran
Date: 10/22/2012
*/

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "../../runopt.h"
#include "../../config.h"

#include "../../Train_factory/Train_factory.h"
#include "../../Test_factory/Test_factory.h"
#include "../../KNN_factory/KNN_factory.h"

#include "AlignParaKNN_GPU.h"


std::string AlignParaKNN_GPU::GetName() {
  return "AlignParaKNN_GPU";
}

AlignParaKNN_GPU::AlignParaKNN_GPU(const Runopt* _options) {
  options = _options;
  Train   = NULL;
  Test = NULL;
  KNN = NULL;
  return;
}

AlignParaKNN_GPU::~AlignParaKNN_GPU() {
  if (Train) delete Train;
  Train = NULL;
  if (Test) delete Test;
  Test = NULL;
  if (KNN) delete KNN;
  KNN = NULL;
}

void AlignParaKNN_GPU::CustomizeFunc() {
#ifdef FINE_DEBUG
  printf("AlignParaKNN_GPU: customizeFunc\n");
#endif
  if(options->new_pFile == 0)
    Train = Train_factory::Get_Train(ALIGNTRAIN_GPU,options);
  else
    Train = Train_factory::Get_Train(ALIGNTRAIN_NEWPFILE_GPU,options);
  
  Test = Test_factory::Get_Test(ALIGNTEST_GPU,options, Train);
  KNN   = KNN_factory::Get_KNN(THRUST_KNN_GPU, options, Train, Test); 
}

void AlignParaKNN_GPU::PreExecution() {
#ifdef FINE_DEBUG
  printf("AlignParaKNN_GPU: PreExecution\n");
#endif
  Train->Setup_Train();
  Test->Setup_Test();
  KNN->Setup_KNN();
  printf("PreExecution Done\n");
}

void AlignParaKNN_GPU::RunBatch() {
#ifdef FINE_DEBUG
  printf("AlignParaKNN_GPU: RunBatch\n");
#endif
  Train->Load_Train();
  Test->Load_Test();
  KNN->AllocateDataStructs();
  printf("Now perform KNN\n");
  KNN->getKNN();
}
