//*******************************************************
// Author: Akshay Chandrashekaran
// Date: 10/16/2012
// Description: Functions to load train data to CPU
//*******************************************************

#include <string>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <assert.h>

#include "../ITrain.h"
#include "AlignTrain_GPU.h"
#include "../../runopt.h"

#define HOFFSET     0  // Header size 
#define OBJSOFFSET  4  // Location of numObjs
#define ROWOFFSET   8  // location of numRows
#define COLOFFSET   12 // location of numCols
#define DATAOFFSET  16 // Offset for data in trainFile

#define LABELOFFSET 8  // location of labels in labelFile

using namespace std;

#define BYTE_4_REVERSE(x)       ((((x)<<24)&0xFF000000U) | (((x)<<8)&0x00FF0000U) | \
				 (((x)>>8) & 0x0000FF00U) | (((x)>>24)&0x000000FFU))
#define BYTE_2_REVERSE(x)       ((((x)<<8)&0xFF00U) | (((x)>>8) & 0x00FFU))

std::string AlignTrain_GPU::GetName() {
  return("AlignTrain_GPU");
}

AlignTrain_GPU::AlignTrain_GPU(const Runopt* _options) {
  options = _options;
  objectData = NULL;
  objLabel = NULL;
  sentenceID = NULL;
  frameID = NULL;
}

AlignTrain_GPU::~AlignTrain_GPU() {
  if (objectData) {
    cudaFree(objectData);   // Free the actual data
  }
  objectData = NULL;
  
  if (objLabel) {
    cudaFree (objLabel);  // Free object Labels
  }
  objLabel = NULL;
  
  if(sentenceID) free(sentenceID);
  sentenceID = NULL;
  
  if (frameID) free(frameID);
  frameID = NULL;
}

void AlignTrain_GPU::Setup_Train() {
  //sleep(10);
  K = options->K;
  numCoords = options->numCoords;
  trainFile = fopen((options->align_train).c_str(), "rb");
  std::cout << (options->align_train).c_str() << " ";
  fseek(trainFile,0,SEEK_END);
  unsigned int size = ftell(trainFile);
  fseek(trainFile,0,SEEK_SET);
  numObjs = (size/4)/(numCoords + 1 + 4); // 1: label(pdfid) 4: uttID + frameID + dummies
  offset = (numCoords + 1 + 4)*4;
  printf("size = %d\n", size);
  //sleep(5);
  fclose(trainFile);
  //sleep(5);
}
// This function will load in the input Train Data and Labels
// It assumes the data to be stored in the MNIST format for now
void AlignTrain_GPU::Load_Train() {
  // Get the training data
  printf("Loading Data\n");
  float *hostObjectData;
  labelType *hostObjLabel;
  trainFile = fopen((options->align_train).c_str(), "rb");
  int featOffset = 4;
  int labelOffset= featOffset + numCoords;
  maxLabel = 0;
  // Get the header data
  float *tempData = (float *) malloc(sizeof(float)*numObjs*numCoords);
  labelType* tempLabel = (labelType *) malloc(sizeof(labelType)*numObjs); 
  sentenceID = (unsigned int* ) malloc(sizeof(unsigned int)*numObjs);
  frameID    = (unsigned int* ) malloc(sizeof(unsigned int)*numObjs);
  
  //Transpose data for convenience
  for (unsigned int i = 0; i < numObjs; i++) {
    
    fseek(trainFile, offset*i, SEEK_SET);
    unsigned int sentence_id;
    fread(&sentence_id,sizeof(unsigned int), 1, trainFile);
    sentenceID[i] = sentence_id;
    
    fseek(trainFile, offset*i + 8, SEEK_SET);
    unsigned int frame_id;
    fread(&frame_id,sizeof(unsigned int), 1, trainFile);
    frameID[i] = frame_id;
    
    featOffset = (offset*i) + (4*4);
    labelOffset =(offset*i) + (4 + numCoords)*4;
    for (unsigned int j = 0; j < numCoords; j++) {
      float tempFloat;
      fseek(trainFile,featOffset, SEEK_SET);
      fread(&tempFloat,4,1,trainFile);
      unsigned char *cptr = reinterpret_cast<unsigned char *>(&tempFloat);
      unsigned char tmp = cptr[0];
      cptr[0] = cptr[3];
      cptr[3] = tmp;
      tmp = cptr[1];
      cptr[1] = cptr[2];
      cptr[2] = tmp;      
      featOffset+= 4;
      tempData[j*numObjs + i] = tempFloat;
      
    }
    labelType tempInt;
    fseek(trainFile, labelOffset, SEEK_SET);
    fread(&tempInt,sizeof(labelType),1,trainFile);
    tempInt = BYTE_4_REVERSE(tempInt);
    if (tempInt > maxLabel)
      maxLabel = tempInt;
    
    tempLabel[i] = tempInt;

  }
  printf("Transposition Done\n");
  
  hostObjectData = tempData;
  hostObjLabel   = tempLabel;
  
  
  
  // Copy training data to GPU global memory
  int memSizeObject = numObjs*numCoords*sizeof(float);
  cudaMalloc((void **)&objectData, memSizeObject);
  assert(cudaMemcpy(objectData,
		    hostObjectData,
		    memSizeObject,
		    cudaMemcpyHostToDevice) 
	 == cudaSuccess);
  
  int memSizeObjLabel = numObjs*sizeof(labelType);
  cudaMalloc((void **)&objLabel, memSizeObjLabel);
  assert(cudaMemcpy(objLabel,
		    hostObjLabel,
		    memSizeObjLabel,
		    cudaMemcpyHostToDevice) 
	 == cudaSuccess);
  //sleep(5);
  fseek(trainFile,0,SEEK_SET);
  fclose(trainFile);
  //sleep(5);
  free(hostObjectData);
  free(hostObjLabel);
  
}

// This function will clear the data stored for the objects and 
// labels in the data struct
void AlignTrain_GPU::Clear_Train() {
  if(objectData) {
    cudaFree(objectData);
  }
  objectData = NULL;
  if(objLabel)
    cudaFree(objLabel);
  objLabel = NULL;
  numObjs = 0;
  numCoords = 0;
  K = 0;
  trainFile = NULL;
  trainLabel= NULL;
}
