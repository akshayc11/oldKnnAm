//  =============================================================
//  HYDRA - Auxiliary Data Structures
//
//    auxModels.h
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
//           Jungsuk Kim (jungsuk.kim@sv.cmu.edu)
//	     look up table(concaternated word symbol) class added
// 
//  ============================================================
#ifndef __PARADECODER_AUXILIARY_MODELS_H__
#define __PARADECODER_AUXILIARY_MODELS_H__

#include <string>
#include <vector>

class PD_WordTable {
 public:
  int nWords;
  std::vector<std::string> table;
  std::vector<int> dupID;

  void read_wordList(const char *WordListName);
  void dump_wordList(const char *WordListName);
  void free_wordList();
};

class PD_TriphoneTable{
  int nTriphones;
  std::vector<std::string> table;
  
  void read_triphoneList(const char *TriphoneListName);
  void dump_triphoneList(const char *TriphoneListName);
  void free_triphoneList();

};

// List of segmentLists
class PD_topList {
 public:
  int fileCount;
  std::vector<std::string> filenames_in;
  std::vector<std::string> filenames_out;
  // Done is 0 if the segment list has not been processed
  // and 1 if it has. Done is updated in WorkerFunction when
  // a thread decides to process its corresponding segment list. 
  std::vector<int> done;

  PD_topList(){
    fileCount = 0;
    filenames_in.clear();
    filenames_out.clear();
    done.clear();
  }

  int read_topList(const char *filename_in, const char *filename_out);
  void free_topList();

};


// List of segment files
class PD_segmentList {
 public:
  int fileCount;
  int currFileID;
  std::vector<std::string> filename;
  std::vector<std::string> session;
  std::vector<int>         index;
  std::vector<float>       startTstamp;
  std::vector<float>       endTstamp;

  PD_segmentList(){
    fileCount = 0;
    currFileID = 0;
    filename.clear();
    session.clear();
    index.clear();
    startTstamp.clear();
    endTstamp.clear();
  }

  int read_segmentList(const char *filename);
  void free_segmentList();
};

#endif
