/****************************************
// kNN factory
// Author: Akshay Chandrashekaran
// Date:10/21/12
// Description: Interface for the kNN classifier
*****************************************/

#ifndef __INTERFACE_PARAKNN_H__
#define __INTERFACE_PARAKNN_H__

#include <string>

#include "../Train_factory/ITrain.h"
#include "../Test_factory/ITest.h"
#include "../KNN_factory/IKNN.h"


#include "../runopt.h"

class IParaKNN {

 public:
  virtual std::string GetName() = 0;
  virtual void CustomizeFunc()  = 0;
  virtual void PreExecution()   = 0;
  virtual void RunBatch()       = 0;
  
  
  ITrain* Train;
  ITest*  Test;
  IKNN*   KNN;
 protected:
  const Runopt* options;
  int Align;
  int Validation;
  int Split;
};

#endif //__INTERFACE_PARAKNN_H__

