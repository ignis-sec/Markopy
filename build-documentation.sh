#!/bin/bash

rm -r docs/*
cd documentation

doxygen Doxyfile-Light
doxygen Doxyfile-Dark