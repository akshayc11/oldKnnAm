/*
AlignParaKNN_CPU.cpp
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

#include "AlignParaKNN_CPU.h"


std::string AlignParaKNN_CPU::GetName() {
  return "AlignParaKNN_CPU";
}


AlignParaKNN_CPU::AlignParaKNN_CPU(const Runopt* _options) {
  options = _options;
  Train   = NULL;
  Test    = NULL;
  KNN     = NULL;
  Align = options->Align;
  Validation = options->Validation;
  Split = options->Split;
  return;
}

AlignParaKNN_CPU::~AlignParaKNN_CPU() {
  if (Train) delete Train;
  if (Test)  delete Test;
  if (KNN)   delete KNN;
  Train = NULL;
  Test  = NULL;
  KNN   = NULL;
}

void AlignParaKNN_CPU::CustomizeFunc() {
  printf("AlignParaKNN_CPU::CustomizeFunc\n");
  Train = Train_factory::Get_Train(ALIGNTRAIN_CPU,options);
  Test  = Test_factory::Get_Test(ALIGNTEST_CPU,options, Train);
  KNN   = KNN_factory::Get_KNN(THRUST_KNN_CPU, options, Train, Test);
}

void AlignParaKNN_CPU::PreExecution() {
  printf("AlignParaKNN_CPU::PreExecution\n");
  Train->Setup_Train();
  Test->Setup_Test();
  KNN->Setup_KNN();
}

void AlignParaKNN_CPU::RunBatch() {
  Train->Load_Train();
  Test->Load_Test();
  KNN->AllocateDataStructs();
  KNN->getKNN();
}

