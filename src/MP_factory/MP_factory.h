/***********************************************************************
MP: MP_factory.h
Author: Akshay Chandrashekaran
Date: 03/05/2013
Description: Functions in KNN
**********************************************************************/
#ifndef __MP_FACTORY_H__
#define __MP_FACTORY_H__

#include "IMP.h"
#include "MP_implementation.h"

#include "../runopt.h"
#include "../KNN_factory/IKNN.h"

class MP_factory {
 public:
  static IMP* Get_MP(MPImplementation modelImpl, const Runopt* options, IKNN* _knn);

};

#endif // __MP_FACTORY_H__
