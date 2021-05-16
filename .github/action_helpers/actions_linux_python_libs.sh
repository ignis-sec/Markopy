#!/bin/bash

# work inside include folder
cd /opt/hostedtoolcache/Python/$1*/x64/include;

echo "Working with python$1"

# Create symlink without "m" in name
[[ -f "python$1" ]] || sudo ln -s python$1m python$1;

# copy libraries and includes to global paths
sudo cp -r /opt/hostedtoolcache/Python/$1*/x64/lib/* /usr/local/lib/;
sudo cp -r /opt/hostedtoolcache/Python/$1*/x64/include/py* /usr/include/;
sudo ln -s /usr/include/$1m /usr/include/$1;

# Create symlink without "m" in name
[[ -f "/usr/local/lib/libpython$1.so.1.0" ]] ||  sudo ln -s /usr/local/lib/libpython$1m.so.1.0 /usr/local/lib/libpython$1.so.1.0;
[[ -f "/usr/local/lib/libpython$1.so" ]] ||  sudo ln -s /usr/local/lib/libpython$1m.so /usr/local/lib/libpython$1.so;
