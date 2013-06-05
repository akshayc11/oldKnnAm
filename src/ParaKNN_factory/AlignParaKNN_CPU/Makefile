
FSTROOT=../..

# Basic directory setup for SDK
# (override directories only if they are not already defined)
SRCDIR     ?= 
ROOTDIR    ?= $(FSTROOT)/..
ROOTBINDIR ?= $(ROOTDIR)/../bin
BINDIR     ?= $(ROOTBINDIR)/$(OSLOWER)
ROOTOBJDIR ?= $(FSTROOT)/obj
LIBDIR     := $(ROOTDIR)/../lib
COMMONDIR  := $(ROOTDIR)/../common
SHAREDDIR  := $(ROOTDIR)/../../shared/

include $(FSTROOT)/common_flags.mk


CCFILES		:= AlignParaKNN_CPU.cpp

HFILES		:= AlignParaKNN_CPU.h


################################################################################
# Set up object files
################################################################################
OBJDIR := ../$(ROOTOBJDIR)/$(LIB_ARCH)/$(BINSUBDIR)
OBJS +=  $(patsubst %.cpp,$(OBJDIR)/%.cpp.o,$(notdir $(CCFILES)))
OBJS +=  $(patsubst %.c,$(OBJDIR)/%.c.o,$(notdir $(CFILES)))
OBJS +=  $(patsubst %.cu,$(OBJDIR)/%.cu.o,$(notdir $(CUFILES)))

################################################################################
# Rules
################################################################################
$(OBJDIR)/%.c.o : $(SRCDIR)%.c $(C_DEPS)  $(HFILES)
	$(VERBOSE)$(CC) $(CFLAGS) -o $@ -c $<

$(OBJDIR)/%.cpp.o : $(SRCDIR)%.cpp $(C_DEPS) $(HFILES)
	$(VERBOSE)$(CXX) $(CXXFLAGS) -o $@ -c $<

all: $(CCFILES) $(OBJS)
