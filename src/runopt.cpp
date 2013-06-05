//======================================================================
// featExtract: CommandLine Interface Parsing
//
//   runopt.cpp
//
//----------------------------------------------------------------------
// Copyright (c) Carnegie Mellon University, 2012
// 
// Description:
//
//  Methods for reading in COmmand Line Options and config files
//
//
// Revisions:
//    v0.1 : 2012/06/16
//           Akshay Chandrashekaran (akshay.chandrashekaran@sv.cmu.edu)
//           Initial Implementation
//
//======================================================================

#include <stdio.h>
#include <stdlib.h>

#include <boost/spirit/core.hpp>
#include <boost/spirit/actor/increment_actor.hpp>
#include <boost/spirit/actor/push_back_actor.hpp>
#include <boost/spirit/utility/confix.hpp>

using namespace boost::spirit;

#include <iostream>
#include <fstream>
#include <cstring>

#include "runopt.h"

using namespace std;

Runopt::Runopt() {
  K                = 5;
  Sigma            = 0.05; 
  GPU_only         = 1;
  FORCE_GPU_ID     = -1;
  MULTI_GPU_IMPL   = 1;
  Align            = 1;
  Validation       = 0;
}

Runopt::~Runopt() {
}

void Runopt::setAndParseMyOptions(int argc, char** argv) {
  
  cmdName = argv[0];
  printf("INFO: Command used [%s]\n", cmdName.c_str());
  
  configPresent = 1;
  config_txt.assign("./default.conf");
  
  parseConfigOptions(argc, argv);
  
  if (configPresent)
    parseConfigFile();
  
  parseCmdLineOptions(argc, argv);
  
  detectGPU();
  gpuIds.clear();
  
  printFinalRunOptions();
  
}

void Runopt::printFinalRunOptions() {
  if (!QUIET) {
    printf("\n");
    printf("==============================================\n");
    printf("\tkNN Parameters\n");
    printf("\tK = %d\n",K);
    printf("\tGPU_only = %d\n",GPU_only);
    printf("\tFORCE_GPU_ID = %d\n",FORCE_GPU_ID);
    printf("\tMULTI_GPU_IMPL = %d\n",MULTI_GPU_IMPL);
    printf("========================================================\n\n");
  }
}

void Runopt::parseConfigFile() {
  std::ifstream wfile(config_txt.c_str());
  std::string line;
  std::string name, str_txt, num_txt;
  FILE *OPT;
  if((OPT=fopen(config_txt.c_str(), "r+"))==NULL) {
    printf("Cannot open configuration file %s\n", config_txt.c_str());
    exit(1);
  }
  fclose(OPT);
  
  // rules for reading Word Table
  rule<phrase_scanner_t> comment_t = ch_p('#') >> (*(~ch_p('\n')));
  rule<phrase_scanner_t> emptyLine_t = (*(space_p));
  rule<phrase_scanner_t> string_t = (+((~ch_p(':'))))[assign_a(name)] >> ch_p(':') >> ch_p('\"') >> (*(~ch_p('\"')))[assign_a(str_txt)] >> ch_p('\"') ;
  rule<phrase_scanner_t> number_t = (+(~ch_p(':')))[assign_a(name)]  >> ch_p(':') >> (*(~ch_p('\n')))[assign_a(num_txt)];
  
  while( std::getline(wfile, line) ){
    // std::cout << "DEBUG: read " << line << std::endl;
    
    if        ( parse(line.c_str(), comment_t   >> end_p, space_p).full ) {
    } else if ( parse(line.c_str(), emptyLine_t >> end_p, space_p).full ) {
    } else if ( parse(line.c_str(), string_t    >> end_p, space_p).full ) {
      // std::cout << "str [" << name << "] [" << str_txt << "]" <<  std::endl;
      if      (strcmp(name.c_str(),"align-train" )     == 0 ) { align_train    = str_txt;}
      else if (strcmp(name.c_str(),"align-test"  )     == 0 ) { align_test     = str_txt;}
      else if (strcmp(name.c_str(),"opFileName"  )     == 0 ) { opFileName     = str_txt;}
      
      else {
	std::cout << "Error: option [" << name << "] not found" << std::endl;
      }
    } else if ( parse(line.c_str(), number_t    >> end_p, space_p).full ) {
      
      if      ( strcmp(name.c_str(), "QUIET")          == 0 ) { QUIET          = atoi(num_txt.c_str());}            
      else if ( strcmp(name.c_str(), "FORCE_GPU_ID")   == 0 ) { FORCE_GPU_ID   = atoi(num_txt.c_str());}
      else if ( strcmp(name.c_str(), "MULTI_GPU_IMPL") == 0 ) { MULTI_GPU_IMPL = atoi(num_txt.c_str());}
      else if ( strcmp(name.c_str(), "K")              == 0 ) { K              = atoi(num_txt.c_str());}
      else if ( strcmp(name.c_str(), "Sigma")          == 0 ) { Sigma          = atof(num_txt.c_str());}
      else if ( strcmp(name.c_str(), "numCoords")      == 0 ) { numCoords      = atoi(num_txt.c_str());}
      
      else {
	std::cout << "ERROR: option [" << name << "] not found. Did you forget the \" \" around a string?" << std::endl;
      }
    } else {
      std::cout << "DEBUG: read [" << line << ']' << std::endl;
    }
  }
}

void Runopt::printOptions() {
  printf("        \n");                     
  printf("        --help               (-h)  \n");
  printf("        \n");                     
  printf("        --quiet              (-q)  \n");
  printf("        --config             (-c) default.conf  \n");
  printf("        \n");
  printf("        --numCoords              39 \n");
  printf("        --align-train       path/filename \n");
  printf("        --align-test        path/filename \n");
  printf("        --opFileName        path/filename \n");
  printf("        --K                  (-k) 11 \n");
  printf("        --Sigma              (-s) 0.05 \n");
  printf("        \n");
  printf("        --force_gpu_ID       (-f) -1 \n");
  printf("\n\n");
}

void Runopt::parseConfigOptions(int argc, char** argv) {
  int pos = 1;
  while(pos < argc) {
    int validArg = 0;
    if (!validArg &&
	((strcmp(argv[pos],"-c") == 0) ||
	 (strcmp(argv[pos],"--config") == 0))) {
      if(pos >= argc-1) {
	printf("ERROR: config has no input\n");
	printOptions();
	exit(1);
      }
      pos++;
      config_txt = argv[pos];
      configPresent = 1;
      if(!QUIET) 
	printf(" Option: --config [%s]\n",config_txt.c_str());
      validArg = 1;
    }
    pos++;
  }
}

void Runopt::parseCmdLineOptions(int argc, char** argv) {
  int pos = 1;
  while (pos < argc) {
    int validArg = 0;
    // Help
    if(!validArg && 
       ((strcmp(argv[pos], "-h") == 0) || 
	(strcmp(argv[pos], "--help") == 0))){
      printOptions();
      exit(0);
    }
    if(!validArg && 
       (strcmp(argv[pos], "-q") == 0)){
      QUIET = 1;	
      validArg = 1;
    }
    //input_file
    if (!validArg &&
	((strcmp(argv[pos],"--align-train") == 0))) {
      if(pos >= argc-1) {
	printf("ERROR: train list has no input\n");
	printOptions();
	exit(1);
      }
      pos++;
      train_file = argv[pos];
      if(!QUIET) 
	printf(" Option: --train-file [%s]\n",config_txt.c_str());
      validArg = 1;
    }
    //test_file
    if (!validArg &&
	((strcmp(argv[pos],"--align-test") == 0))) {
      if(pos >= argc-1) {
	printf("ERROR: test file has no input\n");
	printOptions();
	exit(1);
      }
      pos++;
      test_file = argv[pos];
      if(!QUIET) 
	printf(" Option: --align-test [%s]\n",config_txt.c_str());
      validArg = 1;
    }
    //output file
    if (!validArg &&
	((strcmp(argv[pos],"--opFileName") == 0))) {
      if(pos >= argc-1) {
	printf("ERROR: output file has no input\n");
	printOptions();
	exit(1);
      }
      pos++;
      test_file = argv[pos];
      if(!QUIET) 
	printf(" Option: --opFileName [%s]\n",config_txt.c_str());
      validArg = 1;
    }
    
    //NumCoords
    if (!validArg &&
	((strcmp(argv[pos],"-numCoords") == 0) ||
	 (strcmp(argv[pos],"--numCoords") == 0))) {
      if(pos >= argc-1) {
	printf("ERROR: numCoords has no input\n");
	printOptions();
	exit(1);
      }
      pos++;
      numCoords = atoi(argv[pos]);
      if(!QUIET) 
	printf(" Option: --numCoords [%s]\n",config_txt.c_str());
      validArg = 1;
    }
    
    
    //K
    if (!validArg &&
	((strcmp(argv[pos],"-k") == 0) ||
	 (strcmp(argv[pos],"--K") == 0))) {
      if(pos >= argc-1) {
	printf("ERROR: K has no input\n");
	printOptions();
	exit(1);
      }
      pos++;
      K = atoi(argv[pos]);
      if(!QUIET) 
	printf(" Option: --K [%s]\n",config_txt.c_str());
      validArg = 1;
    }
    
    //Sigma
    if (!validArg &&
	((strcmp(argv[pos],"-s") == 0) ||
	 (strcmp(argv[pos],"--Sigma") == 0))) {
      if(pos >= argc-1) {
	printf("ERROR: Sigma has no input\n");
	printOptions();
	exit(1);
      }
      pos++;
      Sigma = atof(argv[pos]);
      if(!QUIET) 
	printf(" Option: --Sigma [%s]\n",config_txt.c_str());
      validArg = 1;
    }

    pos++;
  }
}
