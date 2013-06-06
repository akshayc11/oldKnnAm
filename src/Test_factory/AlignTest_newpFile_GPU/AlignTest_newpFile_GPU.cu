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
#include "AlignTest_newpFile_GPU.h"

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

std::string AlignTest_newpFile_GPU::GetName() {
  return("AlignTest_newpFile_GPU");
}


AlignTest_newpFile_GPU::AlignTest_newpFile_GPU(const Runopt* _options, ITrain* _train) {
  options = _options;
  objectData = NULL;
  objLabel = NULL;
  sentenceID = NULL;
  frameID = NULL;
  train = _train;
}

AlignTest_newpFile_GPU::~AlignTest_newpFile_GPU() {
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

void AlignTest_newpFile_GPU::Setup_Test() {
  printf("AlignTest_newpFile_GPU::Setup Test\n");
  K = options->K;
    inFile.open((options->align_test).c_str(), std::ios::in | std::ios::binary);
  std::cout << "AlignTest_newpFile_GPU::Setup_Test" << std::endl;
  if (inFile.is_open()) {
    char buf[1024];
    do {
      inFile.getline(buf,1024);
      std::string buffer(buf);
      std::cout << buffer << std::endl;
      if (buffer.find("-end") != std::string::npos)
	break;
      {
	size_t pos = buffer.find("-pfile_header");
	if (pos != std::string::npos) {
	  pos = buffer.find("size ");
	  if (pos == std::string::npos) {
	    std::cerr << "Erroneous header file: Expeced size to be a part of this" << std::endl;
	    exit(1);
	  }
	  else {
	    pos += 5;
	    std::string header_size = buffer.substr(pos);
	    headerSize = atoi(header_size.c_str());
	  }
	}
      }
      {
	size_t pos = buffer.find("-num_frames "); 
	if (pos != std::string::npos) {
	  pos += 12;
	  std::string num_Objs = buffer.substr(pos);
	  numObjs = atoi(num_Objs.c_str());
	}
      }
      {
	size_t pos = buffer.find("-num_features ");
	if (pos != std::string::npos) {
	  pos += 14;
	  std::string num_Coords = buffer.substr(pos);
	  numCoords = atoi(num_Coords.c_str());
	}
      }
	
    } while(1);
    // end of header and begin of data
    // headerSize = inFile.tellg();
    
  }

  std::cout << "numObjs: " << numObjs << " numCoords: " << numCoords << std::endl;
  inFile.close();
  offset = (numCoords + 1 + 2)*4;
  //exit(0);

}
// This function will load in the input Test Data and Labels
void AlignTest_newpFile_GPU::Load_Test() {
  // Get the testing data
  printf("AlignTest_newpFile_GPU::LoadTest\n");
  float* tempData;
  labelType* tempLabel;
  float *hostObjectData;
  // Get the header data
  testFile = fopen((options->align_test).c_str(), "rb");
  
  
  
  int featOffset = 2;
  int labelOffset= featOffset + numCoords;
  // Get the header data
  tempData = (float *)          malloc(sizeof(float)*numObjs*numCoords);
  tempLabel = (labelType *)     malloc(sizeof(labelType)*numObjs); 
  sentenceID = (unsigned int* ) malloc(sizeof(unsigned int)*numObjs);
  frameID    = (unsigned int* ) malloc(sizeof(unsigned int)*numObjs);


  for (unsigned int i = 0; i < numObjs; i++) {
    
    fseek(testFile, headerSize + offset*i, SEEK_SET);
    unsigned int sentence_id;
    fread(&sentence_id,sizeof(unsigned int), 1, testFile);
    sentenceID[i] = sentence_id;
    
    fseek(testFile, headerSize + offset*i + 4, SEEK_SET);
    unsigned int frame_id;
    fread(&frame_id,sizeof(unsigned int), 1, testFile);
    frameID[i] = frame_id;

    featOffset  = headerSize + (offset*i) + (8);
    labelOffset = headerSize + (offset*i) + (2 + numCoords)*4;
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
void AlignTest_newpFile_GPU::Clear_Test() {
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
