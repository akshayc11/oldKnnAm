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
#include "AlignTrain_newpFile_GPU.h"
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

std::string AlignTrain_newpFile_GPU::GetName() {
  return("AlignTrain_newpFile_GPU");
}

AlignTrain_newpFile_GPU::AlignTrain_newpFile_GPU(const Runopt* _options) {
  options = _options;
  objectData = NULL;
  objLabel = NULL;
  sentenceID = NULL;
  frameID = NULL;
}

AlignTrain_newpFile_GPU::~AlignTrain_newpFile_GPU() {
  if (objectData) {
    cudaFree(objectData);   // Free the actual data
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

void AlignTrain_newpFile_GPU::Setup_Train() {
  //sleep(10);
  K = options->K;
  //numCoords = options->numCoords;
  inFile.open((options->align_train).c_str(), std::ios::in | std::ios::binary);
  std::cout << "AlignTrain_newpFile_GPU::Setup_Train" << std::endl;
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
    std::cout << "headerSize: " << headerSize << std::endl;
  }

  std::cout << "numObjs: " << numObjs << " numCoords: " << numCoords << std::endl;
  inFile.close();
  offset = (numCoords + 1 + 2)*4;
}
// This function will load in the input Train Data and Labels
// It assumes the data to be stored in the MNIST format for now
void AlignTrain_newpFile_GPU::Load_Train() {
  // Get the training data
  printf("Loading Data\n");
  float *hostObjectData;
  labelType *hostObjLabel;
  trainFile = fopen((options->align_train).c_str(), "rb");
  //inFile.open((options->align_train).c_str(), std::ios::in | std::ios::binary);
  int featOffset = 2;
  int labelOffset= featOffset + numCoords;
  maxLabel = 0;
  // Get the header data
  float *tempData = (float *) malloc(sizeof(float)*numObjs*numCoords);
  labelType* tempLabel = (labelType *) malloc(sizeof(labelType)*numObjs); 
  sentenceID = (unsigned int* ) malloc(sizeof(unsigned int)*numObjs);
  frameID    = (unsigned int* ) malloc(sizeof(unsigned int)*numObjs);
  
  inFile.open((options->align_train).c_str(), std::ios::in | std::ios::binary);
  
  //Transpose data for convenience
  for (unsigned int i = 0; i < numObjs; i++) {
    
    fseek(trainFile, headerSize + (offset*i), SEEK_SET);
    unsigned int sentence_id;
    
    fread(&sentence_id,sizeof(unsigned int), 1, trainFile);
    sentenceID[i] = sentence_id;
    
    fseek(trainFile, headerSize + (offset*i) + 4, SEEK_SET);
    unsigned int frame_id;
    fread(&frame_id,sizeof(unsigned int), 1, trainFile);
    frameID[i] = frame_id;
    
    featOffset  = headerSize + (offset*i) + 8;
    labelOffset = headerSize + (offset*i) + (2 + numCoords)*4;
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
  objLabel = hostObjLabel;
  //sleep(5);
  fseek(trainFile,0,SEEK_SET);
  fclose(trainFile);
  //sleep(5);
  free(hostObjectData);
  //free(hostObjLabel);
  

}

// This function will clear the data stored for the objects and 
// labels in the data struct
void AlignTrain_newpFile_GPU::Clear_Train() {
  if(objectData) {
    cudaFree(objectData);
  }
  objectData = NULL;
  if(objLabel)
    free(objLabel);
  objLabel = NULL;
  numObjs = 0;
  numCoords = 0;
  K = 0;
  trainFile = NULL;
  trainLabel= NULL;
}
