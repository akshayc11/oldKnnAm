//*******************************************************
// Author: Akshay Chandrashekaran
// Date: 10/16/2012
// Description: Functions to load test data to CPU
//*******************************************************

#include <string>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <assert.h>

#include "../ITest.h"
#include "AlignTest_GPU.h"

#include "../../runopt.h"
#include "../../Train_factory/ITrain.h"

#define HOFFSET     0  // Header size 
#define OBJSOFFSET  4  // Location of numObjs
#define ROWOFFSET   8  // location of numRows
#define COLOFFSET   12 // location of numCols
#define DATAOFFSET  16 // Offset for data in testFile

#define LABELOFFSET 8  // location of labels in labelFile


#define BYTE_4_REVERSE(x)       ((((x)<<24)&0xFF000000U) | (((x)<<8)&0x00FF0000U) | \
				 (((x)>>8) & 0x0000FF00U) | (((x)>>24)&0x000000FFU))
#define BYTE_2_REVERSE(x)       ((((x)<<8)&0xFF00U) | (((x)>>8) & 0x00FFU))

using namespace std;

std::string AlignTest_GPU::GetName() {
  return("AlignTest_GPU");
}


AlignTest_GPU::AlignTest_GPU(const Runopt* _options, ITrain* _train) {
  options = _options;
  objectData = NULL;
  objLabel = NULL;
  sentenceID = NULL;
  frameID = NULL;
  train = _train;
}

AlignTest_GPU::~AlignTest_GPU() {
  if (objectData) {
    free(objectData);   // Free the actual data
  }
  objectData = NULL;
  
  if (objLabel) {
    free (objLabel);  // Free object Labels
  }
  objLabel = NULL;
  
  if(sentenceID) free(sentenceID);
  sentenceID = NULL;
  
  if (frameID) free(frameID);
  frameID = NULL;
}

void AlignTest_GPU::Setup_Test() {
  printf("AlignTest_GPU::Setup Test\n");
  K = options->K;
  numCoords = options->numCoords;
  

}
// This function will load in the input Test Data and Labels
void AlignTest_GPU::Load_Test() {
  // Get the testing data
  printf("AlignTest_GPU::LoadTest\n");
  float* tempData;
  labelType* tempLabel;
  float *hostObjectData;
  // Get the header data
  testFile = fopen((options->align_test).c_str(), "rb");
  fseek(testFile,0,SEEK_SET);
  std::cout << (options->align_test).c_str() << std::endl;
  fseek(testFile,0,SEEK_END);
  unsigned int size = ftell(testFile);
  fseek(testFile,0,SEEK_SET);
  numObjs = (size/4)/(numCoords + 1 + 4); // 1: label(pdfid) 4: uttID + frameID
  offset = (numCoords + 1 + 4)*4;
  std::cout << "size: " << size << std::endl;

  
  
  int featOffset = 4;
  int labelOffset= featOffset + numCoords;
  // Get the header data
  tempData = (float *) malloc(sizeof(float)*numObjs*numCoords);
  tempLabel = (labelType *) malloc(sizeof(labelType)*numObjs); 
  sentenceID = (unsigned int* ) malloc(sizeof(unsigned int)*numObjs);
  frameID    = (unsigned int* ) malloc(sizeof(unsigned int)*numObjs);


  for (unsigned int i = 0; i < numObjs; i++) {
    
    fseek(testFile, offset*i, SEEK_SET);
    unsigned int sentence_id;
    fread(&sentence_id,sizeof(unsigned int), 1, testFile);
    sentenceID[i] = sentence_id;
    
    fseek(testFile, offset*i + 8, SEEK_SET);
    unsigned int frame_id;
    fread(&frame_id,sizeof(unsigned int), 1, testFile);
    frameID[i] = frame_id;

    featOffset = (offset*i) + (4*4);
    labelOffset = (offset*i) + (4 + numCoords)*4;
    for (unsigned int j = 0; j < numCoords; j++) {
      float tempFloat;
      fseek(testFile,featOffset, SEEK_SET);
      fread(&tempFloat,4,1,testFile);
      unsigned char *cptr = reinterpret_cast<unsigned char *>(&tempFloat);
      unsigned char tmp = cptr[0];
      cptr[0] = cptr[3];
      cptr[3] = tmp;
      tmp = cptr[1];
      cptr[1] = cptr[2];
      cptr[2] = tmp;
      
      featOffset+= 4;
      tempData[i*numCoords + j] = tempFloat;
    }
    labelType tempInt;
    fseek(testFile, labelOffset, SEEK_SET);
    fread(&tempInt,sizeof(labelType),1,testFile);
    tempInt = BYTE_4_REVERSE(tempInt);
    
    tempLabel[i] = tempInt;
  }
  //sleep(5);
  fseek(testFile,0,SEEK_SET);
  fclose(testFile);
  //sleep(5);
  
  hostObjectData = tempData;
  objLabel   = tempLabel;
  
  
  
  
  objectData = hostObjectData;

  
}

// This function will clear the data stored for the objects and 
// labels in the data struct
void AlignTest_GPU::Clear_Test() {
  if(objectData) {
    free(objectData);
  }
  objectData = NULL;
  if(objLabel)
    cudaFree(objLabel);
  objLabel = NULL;
  numObjs = 0;
  numCoords = 0;
  K = 0;
  testFile = NULL;
  testLabel= NULL;
}
