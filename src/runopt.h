//======================================================================
// feature Extraction: Command Line Parsing
//
//    runopt.h
//
//----------------------------------------------------------------------
// Copyright (c) Carnegie Mellon Univrsity, 2012 
//
// Description:
//
//   Data structures and methos\ds for reading in command line options
//
//
// Revisions:
//   v0.1 : 2012/06/16
//        Akshay Chandrashekaran (akshay.chandrashekaran.sv.cmu.edu)
//        Initial Implementation
//
//======================================================================
#ifndef __RUNOPT_H__
#define __RUNOPT_H__

#include <string>
#include <vector>

#include "config.h"

using namespace std;

class Runopt {
 public:
  string cmdName;           // Command Used (executable name
  
  int    GPU_only;          // Only executes GPU code
  int    FORCE_GPU_ID;      // Force the use of a particular GPU
  int    MULTI_GPU_IMPL;    // Useing multiple GPUs when available
  
  
  
  // Related to K
  int    K;	/* K in nearest neighbors */
  float Sigma;	/* Scaling factor for log likelihood computation */
  int    Align;	/* Indicates if the file is of align type (precursor to pfile) */
  int    Validation; 		/*  Indicates if data is to be split into test and train */
  int    Split;			/*  Indicate amount of training data split for validation */
  
  int numCoords; 		/* Number of dimensions in a feature */
  
  // Related to the input and output
  string train_file;        // all input files
  string train_label;       // input labels
  string test_file;         // test files
  string test_label;        // test labels
  string align_train;       // Align file for training
  string align_test;        // Align file for testing
  int    one_out_only;      // test with one utterance only
  int    QUIET;             // Quiet execution
  
  vector < int > gpuIds;    // List of usable GPUs for work distribuition
  vector < int > gpuType;   // Type of usable GPUs
  vector < int > gpuMem;    // Memory size of usable GPUs
  
  float  currDeviceMemSize; // Size of current device used in MB
  
  string config_txt;        // Configuration File
  int    configPresent;     // Flag for detection of config File
  std::string opFileName;   /* Output File Name */
  Runopt();
  ~Runopt();
  
  void setAndParseMyOptions(int argc, char** argv);
  void parseConfigOptions  (int argc, char** argv);
  void parseConfigFile();
  void parseCmdLineOptions (int argc, char** argv);
  void printFinalRunOptions();
  void printOptions();
  
#ifdef USE_GPU
  void detectGPU();
#endif

};

#endif
