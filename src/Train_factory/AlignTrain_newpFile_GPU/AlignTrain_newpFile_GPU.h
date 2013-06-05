//*******************************************************
// Author: Akshay Chandrashekaran
// Date: 10/16/2012
// Description: Header for load train data to CPU
//*******************************************************


#ifndef __ALIGNTRAIN_NEWPFILE_GPU_H__
#define __ALIGNTRAIN_NEWPFILE_GPU_H__

#include "../ITrain.h"
#include "../../runopt.h"
#include <string>
#include <fstream>

class AlignTrain_newpFile_GPU : public ITrain {
 public:
  AlignTrain_newpFile_GPU(const Runopt* _options);
  ~AlignTrain_newpFile_GPU();
  
  std::string GetName();
  void Setup_Train();
  void Load_Train();
  void Clear_Train();
  
  // Variables declared in ITrain.h
  // int numCoords;
  // int numObjs;
  // int K;
  // float ** object;
  // labelType *objLabel;
  // labelType *sentenceID;
  // labelType *frameID;
  // Other required Variables
 private:
  std::ifstream inFile;
  size_t headerSize;
  FILE *trainFile;
  FILE *trainLabel;
  float **hostObject;
  labelType* hostObjLabel;
  int offset;
  
};

#endif // __TRAIN_GPU_H__
