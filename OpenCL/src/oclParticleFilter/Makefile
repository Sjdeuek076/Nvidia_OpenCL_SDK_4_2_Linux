################################################################################
#
# Copyright 1993-2011 NVIDIA Corporation.  All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property and 
# proprietary rights in and to this software and related documentation. 
# Any use, reproduction, disclosure, or distribution of this software 
# and related documentation without an express license agreement from
# NVIDIA Corporation is strictly prohibited.
#
# Please refer to the applicable NVIDIA end user license agreement (EULA) 
# associated with this source code for terms and conditions that govern 
# your use of this NVIDIA software.
#
################################################################################
#
# Common build script for OpenCL samples
#
################################################################################

.SUFFIXES : .cl

CUDA_INSTALL_PATH ?= /usr/local/cuda

ifdef cuda-install
	CUDA_INSTALL_PATH := $(cuda-install)
endif

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
# 'linux' is output for Linux system, 'darwin' for OS X
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))
ifneq ($(DARWIN),)
   SNOWLEOPARD = $(strip $(findstring 10.6, $(shell egrep "<string>10\.6" /System/Library/CoreServices/SystemVersion.plist)))
   LION        = $(strip $(findstring 10.7, $(shell egrep "<string>10\.7" /System/Library/CoreServices/SystemVersion.plist)))
endif

# detect if 32 bit or 64 bit system
HP_64 =	$(shell uname -m | grep 64)
OSARCH= $(shell uname -m)

# Basic directory setup for SDK
# (override directories only if they are not already defined)
SRCDIR     ?= 

#/home/adley/Documents/My_WorkPlace/cpp/invofiltering/Nvidia_OpenCL_SDK_4_2_Linux
ROOTDIR    ?= ../../../
#/home/adley/Documents/My_WorkPlace/cpp/invofiltering/Nvidia_OpenCL_SDK_4_2_Linux/OpenCL/src/oclParticleFilter/obj
ROOTOBJDIR ?= obj
#/home/adley/Documents/My_WorkPlace/cpp/invofiltering/Nvidia_OpenCL_SDK_4_2_Linux/shared/lib
LIBDIR     := $(ROOTDIR)/shared/lib/
#/home/adley/Documents/My_WorkPlace/cpp/invofiltering/Nvidia_OpenCL_SDK_4_2_Linux/shared
SHAREDDIR  := $(ROOTDIR)/shared/
#/home/adley/Documents/My_WorkPlace/cpp/invofiltering/Nvidia_OpenCL_SDK_4_2_Linux/OpenCL
OCLROOTDIR := $(ROOTDIR)/OpenCL/
#/home/adley/Documents/My_WorkPlace/cpp/invofiltering/Nvidia_OpenCL_SDK_4_2_Linux/OpenCL/common
OCLCOMMONDIR ?= $(OCLROOTDIR)/common/
#/home/adley/Documents/My_WorkPlace/cpp/invofiltering/Nvidia_OpenCL_SDK_4_2_Linux/OpenCL/bin
OCLBINDIR    ?= $(OCLROOTDIR)/bin/
#/home/adley/Documents/My_WorkPlace/cpp/invofiltering/Nvidia_OpenCL_SDK_4_2_Linux/OpenCL/bin/linux/release
BINDIR       ?= $(OCLBINDIR)/$(OSLOWER)
#/home/adley/Documents/My_WorkPlace/cpp/invofiltering/Nvidia_OpenCL_SDK_4_2_Linux/OpenCL/common/lib
OCLLIBDIR    := $(OCLCOMMONDIR)/lib
#
INCDIR	     ?= .

# Compilers
CXX        := g++
CC         := gcc
LINK       := g++ -fPIC

# Includes
INCLUDES  += -I$(INCDIR) -I$(OCLCOMMONDIR)/inc -I$(SHAREDDIR)/inc

ifeq "$(strip $(HP_64))" ""
	MACHINE := 32
	USRLIBDIR := -L/usr/lib/
else
	MACHINE := 64
	USRLIBDIR := -L/usr/lib64/
endif


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
LIB_ARCH        := $(OSARCH)

# Determining the necessary Cross-Compilation Flags
# 32-bit OS, but we target 64-bit cross compilation
ifeq ($(x86_64),1)
    LIB_ARCH         = x86_64

    ifneq ($(DARWIN),)
         CXX_ARCH_FLAGS += -arch x86_64
    else
         CXX_ARCH_FLAGS += -m64
    endif
else
# 64-bit OS, and we target 32-bit cross compilation
    ifeq ($(i386),1)
        LIB_ARCH         = i386
        ifneq ($(DARWIN),)
            CXX_ARCH_FLAGS += -arch i386
        else
            CXX_ARCH_FLAGS += -m32
        endif
    else
        ifeq "$(strip $(HP_64))" ""
            LIB_ARCH        = i386
            ifneq ($(DARWIN),)
                CXX_ARCH_FLAGS += -arch i386
            else
                CXX_ARCH_FLAGS += -m32
            endif
        else
            LIB_ARCH        = x86_64
            ifneq ($(DARWIN),)
               CXX_ARCH_FLAGS += -arch x86_64
            else
               CXX_ARCH_FLAGS += -m64
            endif
        endif
    endif
endif

# Compiler-specific flags
CXXFLAGS  := $(CXXWARN_FLAGS) $(CXX_ARCH_FLAGS)
CFLAGS    := $(CWARN_FLAGS) $(CXX_ARCH_FLAGS)
LINK      += $(CXX_ARCH_FLAGS)

# Common flags
COMMONFLAGS += $(INCLUDES) -DUNIX

# Add Mac Flags
ifneq ($(DARWIN),)
	COMMONFLAGS += -DMAC
endif

# Debug/release configuration
ifeq ($(dbg),1)
	COMMONFLAGS += -g
	BINSUBDIR   := debug
	LIBSUFFIX   := D
else 
	COMMONFLAGS += -O3 
	BINSUBDIR   := release
	LIBSUFFIX   :=
	CXXFLAGS    += -fno-strict-aliasing
	CFLAGS      += -fno-strict-aliasing
endif


# OpenGL is used or not (if it is used, then it is necessary to include GLEW)
ifeq ($(USEGLLIB),1)

	ifneq ($(DARWIN),)
		OPENGLLIB := -L/System/Library/Frameworks/OpenGL.framework/Libraries -lGL -lGLU $(SHAREDDIR)/lib/$(OSLOWER)/libGLEW.a
	else
		OPENGLLIB := -lGL -lGLU -lX11 -lXmu
		ifeq "$(strip $(HP_64))" ""
			OPENGLLIB += -lGLEW -L/usr/X11R6/lib
		else
			OPENGLLIB += -lGLEW_x86_64 -L/usr/X11R6/lib64
		endif
	endif

	CUBIN_ARCH_FLAG := -m64
endif

ifeq ($(USEGLUT),1)
	ifneq ($(DARWIN),)
		OPENGLLIB += -framework GLUT
		INCLUDES += -I/System/Library/Frameworks/OpenGL.framework/Headers
	else
		OPENGLLIB += -lglut
	endif
endif

# Libs
ifneq ($(DARWIN),)
   LIB       := -L${OCLLIBDIR} -L$(LIBDIR) -L$(SHAREDDIR)/lib/$(OSLOWER) 
   LIB += -framework OpenCL -framework OpenGL ${OPENGLLIB} -framework AppKit ${ATF} ${LIB} 
else
   LIB       := ${USRLIBDIR} -L${OCLLIBDIR} -L$(LIBDIR) -L$(SHAREDDIR)/lib/$(OSLOWER) 
   LIB += -lOpenCL ${OPENGLLIB} ${LIB} 
endif


# Lib/exe configuration
ifneq ($(STATIC_LIB),)
	TARGETDIR := $(OCLLIBDIR)
	TARGET   := $(subst .a,_$(LIB_ARCH)$(LIBSUFFIX).a,$(OCLLIBDIR)/$(STATIC_LIB))
	LINKLINE  = ar qv $(TARGET) $(OBJS) 
else
	LIB += -loclUtil_$(LIB_ARCH)$(LIBSUFFIX) -lshrutil_$(LIB_ARCH)$(LIBSUFFIX)
	TARGETDIR := $(BINDIR)/$(BINSUBDIR)
	TARGET    := $(TARGETDIR)/$(EXECUTABLE)
	LINKLINE  = $(LINK) -o $(TARGET) $(OBJS) $(LIB)
endif

# check if verbose 
ifeq ($(verbose), 1)
	VERBOSE :=
else
	VERBOSE := @
endif

# Add common flags
CXXFLAGS  += $(COMMONFLAGS)
CFLAGS    += $(COMMONFLAGS)


################################################################################
# Set up object files
################################################################################
OBJDIR := $(ROOTOBJDIR)/$(BINSUBDIR)
#OBJS +=  $(patsubst %.cpp,$(OBJDIR)/%.cpp.o,$(notdir $(CCFILES)))
#OBJS +=  $(patsubst %.c,$(OBJDIR)/%.c.o,$(notdir $(CFILES)))

################################################################################
# Rules
################################################################################
compile=g++ -c -o3

# Rules and targets
all: heston_upf

heston_upf: dir $(OBJDIR)/filters.o $(OBJDIR)/powell.o $(OBJDIR)/filter_utils.o $(OBJDIR)/ran2.o $(OBJDIR)/bessik.o $(OBJDIR)/heston_upf.o
	g++ $(OBJDIR)/*.o -o $(OBJDIR)/heston_upf	

$(OBJDIR)/heston_upf.o: heston_upf.cpp
	${compile} heston_upf.cpp -o $(OBJDIR)/heston_upf.o

## related libraries
$(OBJDIR)/filters.o: filters.cpp
	${compile} filters.cpp -o $(OBJDIR)/filters.o

$(OBJDIR)/powell.o: $(OBJDIR)/f1dim.o $(OBJDIR)/linmin.o $(OBJDIR)/mnbrak.o $(OBJDIR)/brent.o $(OBJDIR)/choldc.o recipes/powell.cpp
	${compile} recipes/powell.cpp -o $(OBJDIR)/powell.o

$(OBJDIR)/ran2.o: recipes/ran2.cpp
	${compile} recipes/ran2.cpp -o $(OBJDIR)/ran2.o

$(OBJDIR)/f1dim.o: recipes/f1dim.cpp
	${compile} recipes/f1dim.cpp -o $(OBJDIR)/f1dim.o

$(OBJDIR)/linmin.o: recipes/linmin.cpp
	${compile} recipes/linmin.cpp -o $(OBJDIR)/linmin.o

$(OBJDIR)/mnbrak.o: recipes/mnbrak.cpp
	${compile} recipes/mnbrak.cpp -o $(OBJDIR)/mnbrak.o

$(OBJDIR)/brent.o: recipes/brent.cpp
	${compile} recipes/brent.cpp -o $(OBJDIR)/brent.o

$(OBJDIR)/choldc.o: recipes/choldc.cpp
	${compile} recipes/choldc.cpp -o $(OBJDIR)/choldc.o

$(OBJDIR)/filter_utils.o: $(OBJDIR)/choldc.o $(OBJDIR)/chsone.o $(OBJDIR)/moment.o filter_utils.h filter_utils.cpp
	${compile} filter_utils.cpp -o $(OBJDIR)/filter_utils.o

$(OBJDIR)/chsone.o: $(OBJDIR)/gammq.o recipes/chsone.cpp 
	${compile} recipes/chsone.cpp -o $(OBJDIR)/chsone.o

$(OBJDIR)/moment.o: recipes/moment.cpp
	${compile} recipes/moment.cpp -o $(OBJDIR)/moment.o

$(OBJDIR)/gammq.o: $(OBJDIR)/gser.o $(OBJDIR)/gcf.o recipes/gammq.cpp
	${compile} recipes/gammq.cpp -o $(OBJDIR)/gammq.o

$(OBJDIR)/gser.o: $(OBJDIR)/gammln.o recipes/gser.cpp
	${compile} recipes/gser.cpp -o $(OBJDIR)/gser.o

$(OBJDIR)/gcf.o: $(OBJDIR)/gammln.o recipes/gcf.cpp
	${compile} recipes/gcf.cpp -o $(OBJDIR)/gcf.o

$(OBJDIR)/gammln.o: recipes/gammln.cpp
	${compile} recipes/gammln.cpp -o $(OBJDIR)/gammln.o

$(OBJDIR)/chebev.o: recipes/chebev.cpp
	${compile} recipes/chebev.cpp -o $(OBJDIR)/chebev.o


$(OBJDIR)/beschb.o: recipes/beschb.cpp $(OBJDIR)/chebev.o
	${compile} recipes/beschb.cpp -o $(OBJDIR)/beschb.o

$(OBJDIR)/bessik.o: recipes/bessik.cpp $(OBJDIR)/beschb.o
	${compile} recipes/bessik.cpp -o $(OBJDIR)/bessik.o
	
	
dir:
	mkdir -p obj/release

clean: 
	rm -v ./obj/release/*	

#makedirectories:
#dir :
	#$(VERBOSE)mkdir -p $(LIBDIR)
	#$(VERBOSE)mkdir -p $(OBJDIR)
	#$(VERBOSE)mkdir -p $(TARGETDIR)


#tidy :
	#$(VERBOSE)find . | egrep "#" | xargs rm -f
	#$(VERBOSE)find . | egrep "\~" | xargs rm -f

#clean : tidy
	#$(VERBOSE)rm -f $(OBJS)
	#$(VERBOSE)rm -f $(TARGET)
	#$(VERBOSE)rm -f $(ROOTBINDIR)/$(OSLOWER)/$(BINSUBDIR)/*.ppm
	#$(VERBOSE)rm -f $(ROOTBINDIR)/$(OSLOWER)/$(BINSUBDIR)/*.pgm
	#$(VERBOSE)rm -f $(ROOTBINDIR)/$(OSLOWER)/$(BINSUBDIR)/*.bin
	#$(VERBOSE)rm -f $(ROOTBINDIR)/$(OSLOWER)/$(BINSUBDIR)/*.bmp
	#$(VERBOSE)rm -f $(ROOTBINDIR)/$(OSLOWER)/$(BINSUBDIR)/*.txt
	#$(VERBOSE)rm -f $(LIBDIR)/*.a
	#$(VERBOSE)rm -rf $(ROOTOBJDIR)
	#$(VERBOSE)rm -rf $(OBJDIR)
	#$(VERBOSE)rm -rf $(TARGETDIR)

#clobber : clean
	#$(VERBOSE)rm -f $(TARGETDIR)/samples.list;
	#$(VERBOSE)rm -rf $(COMMONDIR)/lib/*.a
	#$(VERBOSE)rm -rf $(SHAREDDIR)/lib/*.a
	#$(VERBOSE)rm -rf $(COMMONDIR)/obj
	#$(VERBOSE)rm -rf $(SHAREDDIR)/obj









	

	
#dir:
#	mkdir -p bin

#clean: 
#	rm -v ./bin/*
#ekf:
#	g++ ekf.cpp filters.cpp ./recipes/powell.cpp ./recipes/f1dim.cpp ./recipes/linmin.cpp ./recipes/mnbrak.cpp ./recipes/brent.cpp -o ekf	
	
	



