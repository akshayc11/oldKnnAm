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
#include "Test_GPU.h"
#include "../../runopt.h"
#include "../../Train_factory/ITrain.h"

#define HOFFSET     0  // Header size 
#define OBJSOFFSET  4  // Location of numObjs
#define ROWOFFSET   8  // location of numRows
#define COLOFFSET   12 // location of numCols
#define DATAOFFSET  16 // Offset for data in testFile

#define LABELOFFSET 8  // location of labels in labelFile

using namespace std;

std::string Test_GPU::GetName() {
  return("Test_GPU");
}

Test_GPU::Test_GPU(const Runopt* _options) {
  options = _options;
  objectData = NULL;
  objLabel = NULL;
}


Test_GPU::~Test_GPU() {
  if (objectData) {
    cudaFree(objectData);   // Free the actual data
  }
  objectData = NULL;
  
  if (objLabel) {
    free (objLabel);  // Free object Labels
  }
  objLabel = NULL;
}

void Test_GPU::Setup_Test() {
  unsigned char temp[4];
  int header=0;
  unsigned int row = 0;
  unsigned int col = 0;
  K = options->K;

  testFile = fopen((options->test_file).c_str(), "rb");
  printf("%ld\n",ftell(testFile));
  fread(temp, sizeof(int),1,testFile); // Get Header
  header=((((unsigned int)temp[0])<<24) +
	  (((unsigned int)temp[1])<<16) +
	  (((unsigned int)temp[2])<<8) +
	  (((unsigned int)temp[3])));
  printf("header:%x,%ld\n",header,ftell(testFile));
  fread(temp, sizeof(int),1,testFile); // Get numberof Objects
  numObjs=((((unsigned int)(temp[0]&0xff))<<24) +
	   (((unsigned int)temp[1]&0xff)<<16) +
	   (((unsigned int)temp[2]&0xff)<<8) +
	   (((unsigned int)temp[3]&0xff)));
  
  printf("numObjs = %d, %ld\n",numObjs,ftell(testFile));
  fread(temp,sizeof(unsigned int),1,testFile); // Get numberof Objects
  row=((((unsigned int)temp[0])<<24) +
       (((unsigned int)temp[1])<<16) +
       (((unsigned int)temp[2])<<8) +
       (((unsigned int)temp[3])));
  printf("rows = %d, %ld\n",row,ftell(testFile));
  fread(temp,sizeof(unsigned int),1,testFile); // Get numberof Objects
  col=((((unsigned int)temp[0])<<24) +
       (((unsigned int)temp[1])<<16) +
       (((unsigned int)temp[2])<<8) +
       (((unsigned int)temp[3])));
  printf("cols = %d, %ld\n",col,ftell(testFile));
    
  numCoords = row*col;
  printf("Loading Test Data Parameters:\n numObjs:%d, numCoords:%d\n",numObjs, numCoords);
  testLabel = fopen((options->test_label).c_str(), "rb");
  

}
// This function will load in the input Test Data and Labels
// It assumes the data to be stored in the MNIST format for now
void Test_GPU::Load_Test() {
  // Get the testing data
  labelType *data;
  float         *hostObjectData;
  // Get the header data
  
    
    
  data = (labelType *) malloc(sizeof(labelType)*numObjs*numCoords);
  hostObjectData = (float *) malloc (sizeof(float)*numObjs*numCoords);
  fseek(testFile,DATAOFFSET,SEEK_SET);
  fread(data, sizeof(labelType),numObjs*numCoords,testFile);
  printf("Test_GPU: Converting to float\n");
  for (unsigned int i = 0; i < numObjs*numCoords; i++) {
    hostObjectData[i] = (float)(data[i]);
  }
  
  for (unsigned int i = 0; i < numCoords; i++) {
    if (i%28==0)
      printf("\n");
    printf("%0.1f\t", hostObjectData[i]);
  }
  
  // Copy over object data to device global memory
  int memSizeObject = numObjs*numCoords*sizeof(float);
  cudaMalloc((void **)&objectData, memSizeObject);
  cudaMemcpy(objectData,
	     hostObjectData,
	     memSizeObject,
	     cudaMemcpyHostToDevice);
  fclose(testFile);

  free(hostObjectData);
  free(data);
  
  //Get the testing label
  fseek(testLabel,LABELOFFSET,SEEK_SET);
  data = (labelType *)malloc(sizeof(labelType) * numObjs);
  
  fread(data, sizeof(labelType), numObjs, testLabel);
  
  int memSizeObjLabel = numObjs*sizeof(labelType);
  //cudaMalloc((void**)&objLabel, memSizeObjLabel);
  // cudaMemcpy(objLabel,
  // 	     data,
  // 	     memSizeObjLabel,
  // 	     cudaMemcpyHostToDevice);
  
  // free(data);
  objLabel = data;
  fclose(testLabel);
}

// This function will clear the data stored for the objects and 
// labels in the data struct
void Test_GPU::Clear_Test() {
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
  testFile = NULL;
  testLabel= NULL;
}
