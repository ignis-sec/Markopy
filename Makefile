CC		:= g++
C_FLAGS := -Wall -Wextra -g

BIN		:= bin
MP_SRC	:= $(shell find ./MarkovPasswords/src/ -name '*.cpp') $(shell find ./MarkovModel/src/ -name '*.cpp')
MM_SRC  := $(shell find ./MarkovModel/src/ -name '*.cpp')
INCLUDE	:= include
LIB		:= lib

LIBRARIES	:=


MODEL_LIB	:= MarkovModel
PASSWD_EXEC := Markov

all: $(BIN)/$(PASSWD_EXEC)


model: $(BIN)/$(MODEL_LIB)


MarkovPasswords:
	

clean:
	$(RM) $(BIN)/*


$(BIN)/$(PASSWD_EXEC): $(MP_SRC)
		$(CC) $(C_FLAGS) -I$(INCLUDE) -L$(LIB) $^ -o $@ $(LIBRARIES)	

$(BIN)/$(MODEL_LIB): $(MM_SRC)
		$(CC) $(C_FLAGS) -I$(INCLUDE) -L$(LIB) $^ -o $@ $(LIBRARIES)