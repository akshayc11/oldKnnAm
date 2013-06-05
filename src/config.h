//======================================================================
// GPU based feature Extraction
//  config.h
//
//
//----------------------------------------------------------------------
//
// Description:
//
//    Header file for the configuration of the feature extraction Engine
//     - Compile Options
//     - Initialization Parameters
//
// Revisions:
//   v0.1: 2012/07/16
//     Akshay Chandrashekaran (akshay.chandrashekaran@sv.cmu.edu)
//     Initial Implementation
//
//======================================================================

#ifndef __CONFIG_H__
#define __CONFIG_H__

// Enable this selectively for functional checking
//-------------------------------------------------
#define FINE_DEBUG
#define TIME_PROF
#define TIME_RTF

#define USE_GPU 1

// Define the types of GPUs that can run this code
//------------------------------------------------
#define GPU_MAJOR    1
#define GPU_MINOR    2
#define GPU_MEMORY    200000000

// TIMER: must enable for single GPU runs
//             not supported in multiprocessor mode
//-------------------------------------------------

#define TIMING

// Limit the length og input file (in frames)
#define MAX_utterance_length   6000000
#define MAX_IN_SIZE            1500000

// Numerical Accuracy Enhancements
// -Enabled for comparison between CPU and GPU
//-------------------------------------------------
#define KAHANSUM // Numerical accuracy on GPU

// Mathematical Definitions
#define PI    3.1415926535897932384626433832795f
//Speculation Control
//-------------------------------------------------
#define MAX_SPECULATION 32

// label Type
#define labelType int

#endif
