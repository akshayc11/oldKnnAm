# ==============================================================
#
#    Common_flags.mk 
#
# --------------------------------------------------------------
# Copyright (c) Carnegie Mellon University, 2011-2012
#
# Description: 
#
#   Build flags for the Linux Environment
#
# ============================================================

.SUFFIXES : .cu .cu_dbg.o .c_dbg.o .cpp_dbg.o .cu_rel.o .c_rel.o .cpp_rel.o .cubin .ptx

# Add new SM Versions here as devices with new Compute Capability are released
SM_VERSIONS   := 10 11 12 13 20

CUDA_INSTALL_PATH ?= /usr/local/cuda
# CUDA_INSTALL_PATH ?= /usr/local64/lang/cuda-3.2
CUDASDK_INSTALL_PATH ?= /home/akshayc/NVIDIA_GPU_Computing_SDK/C
ifdef cuda-install
	CUDA_INSTALL_PATH := $(cuda-install)
endif

BOOST_INSTALL_PATH ?= /opt/local/include

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

# 'linux' is output for Linux system, 'darwin' for OS X
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))
ifneq ($(DARWIN),)
   SNOWLEOPARD = $(strip $(findstring 10.6, $(shell egrep "<string>10\.6" /System/Library/CoreServices/SystemVersion.plist)))
endif

# detect 32-bit or 64-bit platform
HP_64 = $(shell uname -m | grep 64)
OSARCH= $(shell uname -m)

# Compilers
NVCC       := $(CUDA_INSTALL_PATH)/bin/nvcc -Xptxas -v 
CXX        := g++
CC         := gcc
LINK       := g++ -fPIC

# Includes
INCLUDES  += -I. -I$(CUDA_INSTALL_PATH)/include -I$(CUDASDK_INSTALL_PATH)/common/inc -I$(BOOST_INSTALL_PATH)/ -I$(COMMONDIR)/inc -I$(SHAREDDIR)/inc -I$(FSTROOT)

# Warning flags
CXXWARN_FLAGS := \
	-W -Wall \
	-Wimplicit \
	-Wswitch \
	-Wformat \
	-Wchar-subscripts \
	-Wparentheses \
	-Wmultichar \
	-Wtrigraphs \
	-Wpointer-arith \
	-Wcast-align \
	-Wreturn-type \
	-Wno-unused-function \
	$(SPACE)

CWARN_FLAGS := $(CXXWARN_FLAGS) \
	-Wstrict-prototypes \
	-Wmissing-prototypes \
	-Wmissing-declarations \
	-Wnested-externs \
	-Wmain \

# architecture flag for nvcc and gcc compilers build
CUBIN_ARCH_FLAG :=
CXX_ARCH_FLAGS  :=
NVCCFLAGS       :=
LIB_ARCH        := $(OSARCH)

# Determining the necessary Cross-Compilation Flags
# 32-bit OS, but we target 64-bit cross compilation
ifeq ($(x86_64),1) 
    NVCCFLAGS       += -m64
    LIB_ARCH         = x86_64
    CUDPPLIB_SUFFIX  = x86_64
    ifneq ($(DARWIN),)
         CXX_ARCH_FLAGS += -arch x86_64
    else
         CXX_ARCH_FLAGS += -m64
    endif
else 
# 64-bit OS, and we target 32-bit cross compilation
    ifeq ($(i386),1)
        NVCCFLAGS       += -m32
        LIB_ARCH         = i386
        CUDPPLIB_SUFFIX  = i386
        ifneq ($(DARWIN),)
             CXX_ARCH_FLAGS += -arch i386
        else
             CXX_ARCH_FLAGS += -m32
        endif
    else 
        ifeq "$(strip $(HP_64))" ""
            LIB_ARCH        = i386
            CUDPPLIB_SUFFIX = i386
            NVCCFLAGS      += -m32
            ifneq ($(DARWIN),)
               CXX_ARCH_FLAGS += -arch i386
            else
               CXX_ARCH_FLAGS += -m32
            endif
        else
            LIB_ARCH        = x86_64
            CUDPPLIB_SUFFIX = x86_64
            NVCCFLAGS      += -m64
            ifneq ($(DARWIN),)
               CXX_ARCH_FLAGS += -arch x86_64
            else
               CXX_ARCH_FLAGS += -m64
            endif
        endif
    endif
endif


ifeq ($(noinline),1)
   NVCCFLAGS   += -Xopencc -noinline
   # Compiler-specific flags, when using noinline, we don't build for SM1x
   GENCODE_SM10 := 
   GENCODE_SM20 := -gencode=arch=compute_20,code=\"sm_20,compute_20\"
else
   # Compiler-specific flags (by default, we always use sm_10 and sm_20), unless we use the SMVERSION template
   GENCODE_SM12 := -gencode=arch=compute_12,code=\"sm_12,compute_12\"
   GENCODE_SM13 := -gencode=arch=compute_13,code=\"sm_13,compute_13\"
   GENCODE_SM20 := -gencode=arch=compute_20,code=\"sm_20,compute_20\"
endif

CXXFLAGS  += $(CXXWARN_FLAGS) $(CXX_ARCH_FLAGS)
CFLAGS    += $(CWARN_FLAGS) $(CXX_ARCH_FLAGS)
LINKFLAGS += -lz
POSTLINKFLAGS = -lrt
#-L/usr/local64/lang/cuda-3.2/lib64/
# /n/banquet/da/jike/cuda/lib64/
LINK      += $(LINKFLAGS) $(CXX_ARCH_FLAGS)
POSTLINK  += $(POSTLINKFLAGS)
# This option for Mac allows CUDA applications to work without requiring to set DYLD_LIBRARY_PATH
ifneq ($(DARWIN),)
   LINK += -Xlinker -rpath $(CUDA_INSTALL_PATH)/lib
endif

# Common flags
COMMONFLAGS += $(INCLUDES) -DUNIX

# If we are enabling GPU based debugging, then we want to use -G, warning that this
# May have a significant impact on GPU device code, since optimizations are turned off
ifeq ($(gpudbg),1)
    NVCCFLAGS += -G
	dbg = $(gpudbg)
endif

# Debug/release configuration
ifeq ($(dbg),1)
	COMMONFLAGS += -g
	NVCCFLAGS   += -D_DEBUG -DTHRUST_DEBUG
	CXXFLAGS    += -D_DEBUG
	CFLAGS      += -D_DEBUG
	BINSUBDIR   := debug
	LIBSUFFIX   := D
else 
	COMMONFLAGS += -O2 
	BINSUBDIR   := release
	LIBSUFFIX   := 
	NVCCFLAGS   += --compiler-options -fno-strict-aliasing
	CXXFLAGS    += -fno-strict-aliasing
	CFLAGS      += -fno-strict-aliasing
endif

# architecture flag for cubin build
CUBIN_ARCH_FLAG :=

# OpenGL is used or not (if it is used, then it is necessary to include GLEW)
ifeq ($(USEGLLIB),1)
    ifneq ($(DARWIN),)
        OPENGLLIB := -L/System/Library/Frameworks/OpenGL.framework/Libraries 
        OPENGLLIB += -lGL -lGLU $(COMMONDIR)/lib/$(OSLOWER)/libGLEW.a
    else
# this case for linux platforms
	OPENGLLIB := -lGL -lGLU -lX11 -lXi -lXmu
# check if x86_64 flag has been set, otherwise, check HP_64 is i386/x86_64
        ifeq ($(x86_64),1) 
	       OPENGLLIB += -lGLEW_x86_64 -L/usr/X11R6/lib64
        else
             ifeq ($(i386),)
                 ifeq "$(strip $(HP_64))" ""
	             OPENGLLIB += -lGLEW -L/usr/X11R6/lib
                 else
	             OPENGLLIB += -lGLEW_x86_64 -L/usr/X11R6/lib64
                 endif
             endif
        endif
# check if i386 flag has been set, otehrwise check HP_64 is i386/x86_64
        ifeq ($(i386),1)
	       OPENGLLIB += -lGLEW -L/usr/X11R6/lib
        else
             ifeq ($(x86_64),)
                 ifeq "$(strip $(HP_64))" ""
	             OPENGLLIB += -lGLEW -L/usr/X11R6/lib
                 else
	             OPENGLLIB += -lGLEW_x86_64 -L/usr/X11R6/lib64
                 endif
             endif
        endif
    endif
endif

ifeq ($(USEGLUT),1)
    ifneq ($(DARWIN),)
	OPENGLLIB += -framework GLUT
    else
        ifeq ($(x86_64),1)
	     OPENGLLIB += -lglut -L/usr/lib64 
        endif
        ifeq ($(i386),1)
	     OPENGLLIB += -lglut -L/usr/lib 
        endif

        ifeq ($(x86_64),)
            ifeq ($(i386),)  
	        OPENGLLIB += -lglut
            endif
        endif
    endif
endif

ifeq ($(USEPARAMGL),1)
	PARAMGLLIB := -lparamgl_$(LIB_ARCH)$(LIBSUFFIX)
endif

ifeq ($(USERENDERCHECKGL),1)
	RENDERCHECKGLLIB := -lrendercheckgl_$(LIB_ARCH)$(LIBSUFFIX)
endif

ifeq ($(USECUDPP), 1)
    CUDPPLIB := -lcudpp_$(CUDPPLIB_SUFFIX)$(LIBSUFFIX)

#added by George Caragea, 9/13/10
    INCLUDES += -I$(SHAREDDIR)/inc/cudpp

    ifeq ($(emu), 1)
        CUDPPLIB := $(CUDPPLIB)_emu
    endif
endif

ifeq ($(USENVCUVID), 1)
     ifneq ($(DARWIN),)
         NVCUVIDLIB := -L../../common/lib/darwin -lnvcuvid
     endif
endif

# Libs
ifneq ($(DARWIN),)
    LIB       := -L$(CUDA_INSTALL_PATH)/lib -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER) -L$(SHAREDDIR)/lib $(NVCUVIDLIB) 
else
  ifeq "$(strip $(HP_64))" ""
    ifeq ($(x86_64),1)
       LIB       := -L$(CUDA_INSTALL_PATH)/lib64 -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER) -L$(SHAREDDIR)/lib 
    else
       LIB       := -L$(CUDA_INSTALL_PATH)/lib -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER) -L$(SHAREDDIR)/lib
    endif
  else
    ifeq ($(i386),1)
       LIB       := -L$(CUDA_INSTALL_PATH)/lib -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER) -L$(SHAREDDIR)/lib
    else
       LIB       := -L$(CUDA_INSTALL_PATH)/lib64 -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER) -L$(SHAREDDIR)/lib
    endif
  endif
endif

# If dynamically linking to CUDA and CUDART, we exclude the libraries from the LIB
ifeq ($(USECUDADYNLIB),1)
     LIB += ${OPENGLLIB} $(PARAMGLLIB) $(RENDERCHECKGLLIB) $(CUDPPLIB) ${LIB} -ldl -rdynamic 
else
# static linking, we will statically link against CUDA and CUDART
  ifeq ($(USEDRVAPI),1)
#     LIB += -lcuda   ${OPENGLLIB} $(PARAMGLLIB) $(RENDERCHECKGLLIB) $(CUDPPLIB) ${LIB} 
     LIB += -lcuda   ${OPENGLLIB} $(PARAMGLLIB) $(RENDERCHECKGLLIB) $(CUDPPLIB) ${LIB} 
  else
     ifeq ($(emu),1) 
         LIB += -lcudartemu
     else 
         LIB += -lcudart
     endif
     LIB += ${OPENGLLIB} $(PARAMGLLIB) $(RENDERCHECKGLLIB) $(CUDPPLIB) ${LIB}
  endif
endif

ifeq ($(USECUFFT),1)
  ifeq ($(emu),1)
    LIB += -lcufftemu
  else
    LIB += -lcufft
  endif
endif

ifeq ($(USECUBLAS),1)
  ifeq ($(emu),1)
    LIB += -lcublasemu
  else
    LIB += -lcublas
  endif
endif

# OpenMP is used or not
# Jungsuk Kim added 06/27/2012
#ifeq ($(USEOPENMP),1)
	LIB += -lgomp
	NVCCFLAGS += -Xcompiler -fopenmp
	CXXFLAGS  += -fopenmp
	CFLAGS    += -fopenmp 
#endif

# Lib/exe configuration
ifneq ($(STATIC_LIB),)
	TARGETDIR := $(LIBDIR)
	TARGET   := $(subst .a,_$(LIB_ARCH)$(LIBSUFFIX).a,$(LIBDIR)/$(STATIC_LIB))
# modified by George Caragea, 10/27/10
	LINKLINE  = ar rucv $(TARGET) $(OBJS) $(PROJECT_OBJS)
else
	ifneq ($(OMIT_CUTIL_LIB),1)
		LIB += -lcutil_$(LIB_ARCH)$(LIBSUFFIX) -lshrutil_$(LIB_ARCH)$(LIBSUFFIX)
	endif
	# Device emulation configuration
	ifeq ($(emu), 1)
		NVCCFLAGS   += -deviceemu
		CUDACCFLAGS += 
		BINSUBDIR   := emu$(BINSUBDIR)
		# consistency, makes developing easier
		CXXFLAGS		+= -D__DEVICE_EMULATION__
		CFLAGS			+= -D__DEVICE_EMULATION__
	endif
	TARGETDIR := $(BINDIR)/$(BINSUBDIR)
	TARGET    := $(TARGETDIR)/$(EXECUTABLE)
# modified by George Caragea, 10/27/10
	LINKLINE  = $(LINK) -o $(TARGET) $(OBJS) $(LIB) $(PROJECT_OBJS) -lcuda $(POSTLINK)
endif

# check if verbose 
ifeq ($(verbose), 1)
	VERBOSE :=
else
	VERBOSE := @
endif

################################################################################
# Check for input flags and set compiler flags appropriately
################################################################################
ifeq ($(fastmath), 1)
	NVCCFLAGS += -use_fast_math
endif

ifeq ($(nvccverbose), 1)
        NVCCFLAGS += -v
endif

ifeq ($(keep), 1)
	NVCCFLAGS += -keep
	NVCC_KEEP_CLEAN := *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx
endif

ifdef maxregisters
	NVCCFLAGS += -maxrregcount $(maxregisters)
endif

# Add cudacc flags
NVCCFLAGS += $(CUDACCFLAGS)

# Add common flags
NVCCFLAGS += $(COMMONFLAGS)
CXXFLAGS  += $(COMMONFLAGS)
CFLAGS    += $(COMMONFLAGS)

ifeq ($(nvcc_warn_verbose),1)
	NVCCFLAGS += $(addprefix --compiler-options ,$(CXXWARN_FLAGS)) 
	NVCCFLAGS += --compiler-options -fno-strict-aliasing
endif
