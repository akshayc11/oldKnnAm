/***********************************************************************
MP: MP_factory.cpp
Author:/Akshay Chandrashekaran
Date: 03/05/2013
Description: Factory Pattern for Measure Propagation
**********************************************************************/

#include <assert.h>

#include "../runopt.h"
#include "MP_implementation.h"
#include "MP_factory.h"

#include "MP_GPU/MP_GPU.h"

#include "../KNN_factory/IKNN.h"

#include <iostream>

using namespace std;

IMP* MP_factory::Get_MP(MP_implementation modelImpl, const Runopt* options, IKNN* KNN) {
  switch (modelImpl) {
  case(MP_GPU): return new MP_GPU(options, KNN);
  default: cerr << "Error: MP factory cant find the specified model implementation" << endl; 
  }
  return 0;
}
