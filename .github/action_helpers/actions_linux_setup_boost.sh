#!/bin/bash

wget https://boostorg.jfrog.io/artifactory/main/release/1.76.0/source/boost_1_76_0.tar.gz -O /home/runner/boost_1_76_0.tar.gz -q;
cd /home/runner/;
tar -xf /home/runner/boost_1_76_0.tar.gz;
cd /home/runner/boost_1_76_0/;
./bootstrap.sh --with-python=$(which python$1) --with-python-version=$1;
./b2 --with-python --with-program_options stage;