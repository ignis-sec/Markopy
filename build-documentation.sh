#!/bin/bash

export CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH:+${CPLUS_INCLUDE_PATH}:}/usr/include/python3.8:/usr/lib/cuda/include:$(pwd)"
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/lib
#cleanup old documentations
rm -r docs/*
rm -r documentation/latex/*

cd documentation

# Light theme documentation
cat Doxyfile-Light.comp > Doxyfile-Light
cat Doxyfile-Base.comp >> Doxyfile-Light
doxygen Doxyfile-Light

# Dark theme documentation
cat Doxyfile-Dark.comp > Doxyfile-Dark
cat Doxyfile-Base.comp >> Doxyfile-Dark
doxygen Doxyfile-Dark

# Dark theme documentation
cat Doxyfile-Latex.comp > Doxyfile-Latex
cat Doxyfile-Base.comp >> Doxyfile-Latex
doxygen Doxyfile-Latex

cp -r includes/latex/* latex/
cd latex
make
cp refman.pdf ../../docs/documentation.pdf