# ==============================================================
#
#    Makefile (Source Top Level)
#
# --------------------------------------------------------------
# Copyright (c) Carnegie Mellon University, 2012-
#
# Description: 
#
#   Makefile rules to compile the project under Linux
#
# Revisions:
#
#  ============================================================

EXECUTABLE	:= distKNN.linux

# Cuda source files (compiled with nvcc)
CUFILES := runopt.cu
#CUFILES_sm_13	:= 
#		common_cuda.cu common_llc.cu 

# C++ source files (compiled with gcc)
CCFILES		:= main.cpp runopt.cpp

#added by George Caragea, 10/27/10
RECURSIVE_OBJS	:= common.cpp.o common_cuda.cu.o\
	Train_factory.cpp.o \
	AlignTrain_GPU.cu.o \
	AlignTrain_newpFile_GPU.cu.o \
	Test_factory.cpp.o \
	AlignTest_GPU.cu.o \
	ParaKNN_factory.cpp.o \
	AlignParaKNN_GPU.cpp.o \
	KNN_factory.cpp.o \
	thrustKNN_GPU.cu_20.o \
	thrustKNN_GPU_kernel.cu_20.o \



DIRS		:= shared \
	Train_factory \
	Train_factory/AlignTrain_GPU \
	Test_factory \
	Test_factory/AlignTest_GPU \
	ParaKNN_factory \
	ParaKNN_factory/AlignParaKNN_GPU \
	KNN_factory \
	KNN_factory/thrustKNN_GPU \



#USECUDPP	:= 1

#USEDRVAPI	:= 1

################################################################################
# Rules and targets

FSTROOT		:= .
#ROOTDIR		:= /home/$(LOGNAME)/NVIDIA_GPU_Computing_SDK/C
include ROOTDIR.mk
include common_cuda.mk

MAKE=make
SHELL=/bin/bash

.PHONY: recurse clean $(DIRS)

recurse: $(DIRS)

$(DIRS): 
	@if [[ -d $@ ]]; then \
		echo Running make in $@; \
		cd $@; \
		$(MAKE); \
	 else \
		echo Directory not found: $@. Skipping...; \
	 fi;

################################################################################
# Recursive compilation
#DIRS=ObsModel_factory

#-include $(FSTROOT)/dirs.mk
