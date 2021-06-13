

##############################################################################################################
#####################################              Global Options            #################################
##############################################################################################################

# Compiler
CC            	:= g++
#output directory
BIN            	:= bin
#include directory
INCLUDE        	:= include
#Libraries
LIB            	:= lib
LIBRARIES    	:=

ifndef PYTHON_VERSION
PYTHON_VERSION := $(shell python3 -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)";) #3.8
endif

PYTHON_VERSION_ :=$(shell python$(PYTHON_VERSION) -c "import sys;t='{v[0]}{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)";) #38
PYTHON_VERSION3 :=$(shell python$(PYTHON_VERSION) -c "import sys;t='{v[0]}.{v[1]}.{v[2]}'.format(v=list(sys.version_info[:3]));sys.stdout.write(t)";) #3.8.s2

##############################################################################################################
#####################################     MarkovPassword project options     #################################
##############################################################################################################

MP_C_FLAGS  := -Wall -Wextra -g -Ofast -std=c++17
MP_EXEC     := Markov
MP_SRC      := $(shell find ./MarkovPasswords/src/ -name '*.cpp') 
MP_INC 		:= 
MP_LIB		:= -lboost_program_options -lpthread
MP_INC		:= $(shell pwd)

#build pattern
$(BIN)/$(MP_EXEC): $(MP_SRC)
	$(CC) $(MP_C_FLAGS) -I$(MP_INC) -L$(LIB) $^ -o $@ $(MP_LIB) 

##############################################################################################################
#####################################       MarkovModel project options      #################################
##############################################################################################################
MM_SRC_DIR      := MarkovModel/src/
MM_SRC          := $(shell find $(MM_SRC_DIR) -name '*.cpp')
MM_OBJS         := $(MM_SRC:%=$(BIN)/%.o)
MM_DEPS         := $(MM_OBJS:.o=.d)
MM_LDFLAGS      := -shared 
MM_C_FLAGS      := $(MM_INC_FLAGS) -MMD -MP  -Ofast -std=c++17
MM_INC_DIRS     := $(shell find $(MM_SRC_DIR) -type d)
MM_INC_FLAGS    := $(addprefix -I,$(MM_INC_DIRS))
MM_LIB          := model.so

$(INCLUDE)/$(MM_LIB): $(MM_OBJS)
	echo $(MM_OBJS)
	$(CC) $(MM_OBJS) -o $@ $(MM_LDFLAGS)

# Build step for C++ source
$(BIN)/%.cpp.o:%.cpp
	mkdir -p $(dir $@)
	$(CC) $(MM_C_FLAGS) -c $< -o $@

-include $(MM_DEPS)

##############################################################################################################
#####################################            Markopy Options             #################################
##############################################################################################################

MPY_SRC          := MarkovPasswords/src/markovPasswords.cpp MarkovPasswords/src/threadSharedListHandler.cpp $(shell find Markopy/src/Module/ -name '*.cpp')
MPY_SRC_DIR		 := Markopy/src/
MPY_OBJS         := $(MPY_SRC:%=$(BIN)/%.o)
MPY_DEPS         := $(MPY_OBJS:.o=.d)
MPY_LDFLAGS      := -shared -lboost_python$(PYTHON_VERSION_) -lpython$(PYTHON_VERSION) -lpthread
MPY_C_FLAGS      := $(MPY_INC_FLAGS) -MMD -MP -fPIC -I/usr/include/python$(PYTHON_VERSION)  -Ofast -std=c++17
MPY_INC_DIRS     := $(shell find $(MPY_SRC_DIR) -type d) $(shell pwd)
MPY_INC_FLAGS    := $(addprefix -I,$(MPY_INC_DIRS))
MPY_SO           := markopy.so

$(BIN)/$(MPY_SO): $(MPY_OBJS)
	$(CC) $(MPY_OBJS) -o $@ $(MPY_LDFLAGS)

# Build step for C++ source
$(BIN)/%.cpp.o:%.cpp
	mkdir -p $(dir $@)
	$(CC) $(MPY_C_FLAGS) $(MPY_INC_FLAGS) -c $< -o $@

-include $(MPY_DEPS)

##############################################################################################################
#####################################               Directives               #################################
##############################################################################################################

.PHONY: all
all: model mp markopy
model: $(INCLUDE)/$(MM_LIB)

mp: $(BIN)/$(MP_EXEC)

markopy: $(BIN)/$(MPY_SO)

.PHONY: clean
clean:
	$(RM) -r $(BIN)/*


