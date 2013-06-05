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
#include "AlignTest_CPU.h"

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


std::string AlignTest_CPU::GetName() {
  return("AlignTest_CPU");
}


AlignTest_CPU::AlignTest_CPU(const Runopt* _options, ITrain* _train) {
  options = _options;
  objectData = NULL;
  objLabel = NULL;
  train = _train;
}

AlignTest_CPU::~AlignTest_CPU() {
  if (objectData) {
    free(objectData);   // Free the actual data
  }
  objectData = NULL;
  
  if (objLabel) {
    free (objLabel);  // Free object Labels
  }
  objLabel = NULL;
}

void AlignTest_CPU::Setup_Test() {
  labelType temp[4];
  int header=0;
  unsigned int row = 0;
  unsigned int col = 0;
  K = options->K;
  Validation = options->Validation;
  Align = options->Align;
  Split = options->Split;

}
// This function will load in the input Test Data and Labels
// It assumes the data to be stored in the MNIST format for now
void AlignTest_CPU::Load_Test() {
  // Get the testing data
  
  float* tempData;
  labelType* tempLabel;
  if (Validation == 0) {
    testFile = fopen((options->test_file).c_str(), "rb");
    fseek(testFile,0,SEEK_END);
    unsigned int size = ftell(testFile);
    fseek(testFile,0,SEEK_SET);
    numObjs = (size/4)/(numCoords + 1 + 4); // 1: label(pdfid) 4: uttID + frameID
    offset = (numCoords + 1 + 4)*4;
    //printf("numObjs = %d\n", numObjs);
  }
  else {
    numCoords = train->numCoords;
    numObjs   = train->validNumObjs;
    
  }
    
  if (Validation == 0) {
  
    int featOffset = 4;
    int labelOffset= featOffset + numCoords;
    // Get the header data
    tempData = (float *) malloc(sizeof(float)*numObjs*numCoords);
    tempLabel = (labelType *) malloc(sizeof(labelType)*numObjs); 
    for (unsigned int i = 0; i < numObjs; i++) {
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
    fclose(testFile);
  }
  else {
    numObjs   = train->validNumObjs;
    tempData  = train->validTest;
    tempLabel = train->validLabel;
    
  }
  objectData = tempData;
  objLabel   = tempLabel;
  
}

// This function will clear the data stored for the objects and 
// labels in the data struct
void AlignTest_CPU::Clear_Test() {
  if(objectData) {
    free(objectData);
  }
  objectData = NULL;
  if(objLabel)
    free(objLabel);
  objLabel = NULL;
  numObjs = 0;
  numCoords = 0;
  K = 0;
  testFile = NULL;
  testLabel= NULL;
}
