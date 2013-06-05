//======================================================================
//
//  featExtract
//
//            main.cpp
//----------------------------------------------------------------------
// Copyright (c) CArnegie Mellon University, 2012
//
//  Description:
//    Top level module for the kNN toolkit
//
//  Revisions:
//    v0.1:2012/07/17
//       Akshay Chandrashekaran ( akshay.chandrashekaran@sv.cmu.edu )
//
//=====================================================================

#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include "runopt.h"
#include "ParaKNN_factory/ParaKNN_factory.h"

int main( int argc, char** argv) {
  
  Runopt runopt;
  printf("Parsing Options\n");
  runopt.setAndParseMyOptions(argc, argv);
  printf("Parsed Options\n"); 
  IParaKNN* ParaKNN;
  ParaKNN = ParaKNN_factory::Get_ParaKNN(ALIGNPARAKNN_GPU, &runopt);
  ParaKNN->CustomizeFunc();
  ParaKNN->PreExecution();
  ParaKNN->RunBatch();
  return (0);
}
