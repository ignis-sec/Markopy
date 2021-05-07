

##############################################################################################################
#####################################              Global Options            #################################
##############################################################################################################

# Compiler
CC            := g++
#output directory
BIN            := bin
#include directory
INCLUDE        := include
#Libraries
LIB            := lib
LIBRARIES    :=

##############################################################################################################
#####################################     MarkovPassword project options     #################################
##############################################################################################################

MP_C_FLAGS  := -Wall -Wextra -g
MP_EXEC     := Markov
MP_SRC      := $(shell find ./MarkovPasswords/src/ -name '*.cpp') $(shell find ./MarkovModel/src/ -name '*.cpp')

#build pattern
$(BIN)/$(MP_EXEC): $(MP_SRC)
	$(CC) $(MP_C_FLAGS) -I$(INCLUDE) -L$(LIB) $^ -o $@ $(LIBRARIES)    


##############################################################################################################
#####################################     MarkovPassword project options     #################################
##############################################################################################################

MP_C_FLAGS  := -Wall -Wextra -g
MP_EXEC     := Markov
MP_SRC      := $(shell find ./MarkovPasswords/src/ -name '*.cpp') $(shell find ./MarkovModel/src/ -name '*.cpp')

#build pattern
$(BIN)/$(MP_EXEC): $(MP_SRC)
	$(CC) $(MP_C_FLAGS) -I$(INCLUDE) -L$(LIB) $^ -o $@ $(LIBRARIES)    




##############################################################################################################
#####################################       MarkovModel project options      #################################
##############################################################################################################
MM_SRC_DIR      := MarkovModel/src/
MM_SRC          := $(shell find $(MM_SRC_DIR) -name '*.cpp')
MM_OBJS         := $(MM_SRC:%=$(BIN)/%.o)
MM_DEPS         := $(MM_OBJS:.o=.d)
MM_LDFLAGS      := -shared 
MM_C_FLAGS      := $(MM_INC_FLAGS) -MMD -MP
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

MPY_C_FLAGS  := -c -FPIC
MPY_EXEC     := Markov
MPY_SRC      := Markopy/src/Module/markopy.cpp
MPY_INCLUDE  := 
#build pattern
$(BIN)/$(MP_EXEC): $(MP_SRC)
	$(CC) $(MP_C_FLAGS) -I$(MPY_INCLUDE) -L$(LIB) $^ -o $@  


##############################################################################################################
#####################################               Directives               #################################
##############################################################################################################

.PHONY: all
all: model mp
model: $(INCLUDE)/$(MM_LIB)

mp: $(BIN)/$(MP_EXEC)

.PHONY: clean
clean:
	$(RM) -r $(BIN)/*


