//  =============================================================
//  HYDRA - Auxiliary Data Structures
//
//    auxModels.cpp
//
//  -------------------------------------------------------------
//  CMU, 2011-
// 
//  Description:
// 
//    Auxilary data structures for decoding
//  
//  Revisions:
//    v0.1 : 2012/02/11
//	     Jungsuk Kim (jungsuk.kim@sv.cmu.edu)
//	     look up table(concaternated word symbol) class added
// 
//  ============================================================
#include <stdio.h>
#include <stdlib.h>

#include "auxModels.h"
#include "../config.h"
//#include <boost/spirit/include/classic_core.hpp>
//#include <boost/spirit/include/classic_increment_actor.hpp>
//#include <boost/spirit/include/classic_push_back_actor.hpp>
//#include <boost/spirit/include/classic_confix.hpp
#include <boost/spirit/core.hpp>
#include <boost/spirit/actor/increment_actor.hpp>
#include <boost/spirit/actor/push_back_actor.hpp>
#include <boost/spirit/utility/confix.hpp>
using namespace boost::spirit;

#include <vector>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <float.h>
#include <string.h>

void PD_WordTable::read_wordList(const char *WordListName){

  // std::cout << "\n Read in word list " << WordListName << std::endl;
  std::ifstream wfile(WordListName);
  std::string line;
   
  std::string name;
  int    count = 0;
  int    totalWord;
 
  // rules for reading Word Table
  rule<phrase_scanner_t> wordcount_t = int_p[assign_a(totalWord)];  
  rule<phrase_scanner_t> wordLine_t = (+(~ch_p('\n')))[assign_a(name)];
 
  // Reading word files
  (table).clear();
   
  std::getline(wfile, line);
  if( parse(line.c_str(), wordcount_t >> end_p, space_p).full ) {
  } else {
    std::cout << "ERROR: Word count not recognized " << line << std::endl;
  }
   
  while( std::getline(wfile, line) ){
    // std::cout << "DEBUG: read " << line << std::endl;
    if( parse(line.c_str(), wordLine_t >> end_p, space_p).full ) {
      table.push_back(name);
      // std::cout << "DEBUG: read " << name << std::endl;
      count++;
    } else {
      std::cout << "ERROR: Line not recognized " << line << std::endl;
    }
  }
  if (count == totalWord){
    std::cout << "INFO: all "<< totalWord << " words read in to list." << std::endl;
    nWords = totalWord;
  } else {
    std::cout << "ERROR: Read in word list " << WordListName << std::endl;
    std::cout << "ERROR: expecting "<< totalWord << " words, read in " << count << " to list." << std::endl;
  }
  wfile.close();

  dupID.resize(table.size());
  dupID[0] = 0;
  for(int i=1; i<table.size(); i++){
    if(table[i-1] == table[i]){
      dupID[i] = dupID[i-1]+1;
    } else {
      dupID[i] = 0;
    }
  }
}

// ------------------------------------------------------------------------------
// Dump a list of words into file for verification
//
void PD_WordTable::dump_wordList(const char *WordListName){
  std::ofstream wfile(WordListName);

  wfile << nWords << std::endl;

  for(int i=0; i<nWords; i++){
    wfile << table[i] << std::endl;
  }
  wfile.close();
  std::cout << "DEBUG: Dumped out word list " << WordListName << std::endl;
}

void PD_WordTable::free_wordList(){
  table.clear();
  nWords = 0;
}


// ------------------------------------------------------------------------------
// Read a list of triphones into internal data structure
//
void PD_TriphoneTable::read_triphoneList(const char *TriphoneListName){

  // std::cout << "\n Read in word list " << WordListName << std::endl;
  std::ifstream wfile(TriphoneListName);
  std::string line;
   
  std::string name;
  int    count = 0;
  int    totalTriphone;
 
  // rules for reading Word Table
  rule<phrase_scanner_t> triphonecount_t = int_p[assign_a(totalTriphone)];  
  rule<phrase_scanner_t> triphoneLine_t = (+(~ch_p('\n')))[assign_a(name)];
 
  // Reading word files
  (table).clear();
   
  std::getline(wfile, line);
  if( parse(line.c_str(), triphonecount_t >> end_p, space_p).full ) {
  } else {
    std::cout << "ERROR: Triphone count not recognized " << line << std::endl;
  }
   
  while( std::getline(wfile, line) ){
    // std::cout << "DEBUG: read " << line << std::endl;
    if( parse(line.c_str(), triphoneLine_t >> end_p, space_p).full ) {
      table.push_back(name);
      // std::cout << "DEBUG: read " << name << std::endl;
      count++;
    } else {
      std::cout << "ERROR: Line not recognized " << line << std::endl;
    }
  }

  if (count == totalTriphone){
    // std::cout << "INFO: all "<< totalTriphone << " triphones read in to list." << std::endl;
    nTriphones = totalTriphone;
  } else {
    std::cout << "ERROR: Read in triphone list " << TriphoneListName << std::endl;
    std::cout << "ERROR: expecting "<< totalTriphone << " triphones, read in " << count
              << " to list." << std::endl;
  }
  wfile.close();
}


// ------------------------------------------------------------------------------
// Dump a list of triphone names into file for verification
//
void PD_TriphoneTable::dump_triphoneList(const char *TriphoneListName){
  std::ofstream wfile(TriphoneListName);

  wfile << nTriphones << std::endl;

  for(int i=0; i<nTriphones; i++){
    wfile << table[i] << std::endl;
  }
  wfile.close();
  std::cout << "DEBUG: Dumped out triphone list " << TriphoneListName << std::endl;
}

void PD_TriphoneTable::free_triphoneList(){
  table.clear();
  nTriphones = 0;
}


// ------------------------------------------------------------------------------
// Opening a top-level "list of lists". This file contains one entry for
// every file list of extracted input speech features. 
// This function also process the filenames for the outputs
int PD_topList::read_topList(const char *filename_in, const char *filename_out) {
  
  int fileCount_in = 0;
  int fileCount_out = 0;
        
  std::cout << "\nINFO: Read top input file list of segment lists: " << filename_in << std::endl;
  std::ifstream file_in(filename_in);
  if (file_in.is_open()){
    std::cout << "INFO: Top input file " << filename_in << " successfully opened" << std::endl;
  } else {
    std::cout << "ERROR: Error opening file " << filename_in << std::endl;
  }

  std::cout << "\nINFO: Read top output file list of segment lists: " << filename_out << std::endl;
  std::ifstream file_out(filename_out);
  if (file_out.is_open()){
    std::cout << "INFO: Top output file " << filename_out << " successfully opened" << std::endl;
  } else {
    std::cout << "ERROR: Error opening file " << filename_out << std::endl;
  }

  std::string line;
  int ERROR_FLAG = 0;

  // Read in all lines (each line has an input filename)
  while( std::getline(file_in, line) ){
    // Check if file is available
    std::ifstream filestr(line.c_str());
    if (filestr.is_open()){
      filestr.close();
      // Add this file to the list
      (filenames_in).push_back(line);
      // Initially set gpu ID to 0
      (done).push_back(0);
      fileCount_in++;
    } else {
      ERROR_FLAG = 1;
      std::cout << "ERROR: while opening file " << line << std::endl; ;
    }
  }
  file_in.close();

  // Read in all OUTPUT filename lines (each line has a filename)
  while( std::getline(file_out, line) ){
    (filenames_out).push_back(line);
    fileCount_out++;
  }
  file_out.close();
  if(fileCount_in != fileCount_out)
    {
      printf("fileCount_in %d   fileCount_out %d\n", fileCount_in, fileCount_out);
      ERROR_FLAG = 1;
    }
  if (ERROR_FLAG) { std::cout << "ERROR: opening some segment files\n"; }
  else { std::cout << "INFO: Successfully found all segment files\n";}

  return fileCount_in;
}

void PD_topList::free_topList(){
  filenames_in.clear();
  filenames_out.clear();
  done.clear();
  fileCount = 0;
}

// ------------------------------------------------------------------------------
// Opening a file list for extracted input speech features
// - expecting the following format:
//   ./test/features/58k_1/NIST_20051104-1515_h01_125_1_1768990_1779910.wav!NIST_20051104-1515_h01_125!1 1768.99 1779.91
// - separate out "SessionName", "SessionNr","startTimeOffset","endTimeOffset"
// 
int PD_segmentList::read_segmentList(const char *my_filename){
  //std::string tempWord;
  fileCount = 0;
  std::cout << "\nINFO: Read file List file: " << my_filename << std::endl;
  std::ifstream file(my_filename);
  if (file.is_open()){
    std::cout << "INFO: File list " << my_filename << " successfully opened " << std::endl;
  } else {
    std::cout << "ERROR: Error opening file " << my_filename << std::endl;
  }
  std::string line;

  int ERROR_FLAG = 0;

  
  std::string my_name;
  std::string my_session;
  int my_index;
  float my_startTime;
  float my_endTime;
  rule<phrase_scanner_t> fileline_t = (+(~ch_p('!')))[assign_a(my_name)] >> ch_p('!') >>
    (+(~ch_p('!')))[assign_a(my_session)] >> ch_p('!') >>
    int_p[assign_a(my_index)] >> real_p[assign_a(my_startTime)] >> real_p[assign_a(my_endTime)];


  while( std::getline(file, line) ){
    if( parse(line.c_str(), fileline_t >> end_p, space_p).full ) {
      
      // check if file is available
      std::ifstream filestr(my_name.c_str());
      if (filestr.is_open()){
        //std::cout << "File " << name << " successfully openned " << std::endl;
        filestr.close();
        
        (filename).push_back(my_name);
        (session).push_back(my_session);
        (index).push_back(my_index);
        (startTstamp).push_back(my_startTime);
        (endTstamp).push_back(my_endTime);


        // printf(" %s | %s | %d | %f | %f \n",
        //        (filename)[fileCount],
        //        (session)[fileCount],
        //        (index)[fileCount],
        //        (startTstamp)[fileCount],
        //        (endTstamp)[fileCount]);
    
        fileCount++;
      } else {
        ERROR_FLAG = 1;
        std::cout << "ERROR: while opening file " << my_name << std::endl; ;
      }
    } else {
      std::cout << "ERROR: unexpected line format" << line << std::endl; ;
    }
  }
  file.close();
  
  if (ERROR_FLAG) { std::cout << "ERROR: openning some files\n"; }
  else { std::cout << "INFO: Successfully found all inputs\n";}
  return fileCount;
} 

void PD_segmentList::free_segmentList(){
  fileCount = 0;
  filename.clear();
  session.clear();
  index.clear();
  startTstamp.clear();
  endTstamp.clear();
}
