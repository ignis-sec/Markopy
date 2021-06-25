#!/bin/bash

pwd
whoami
ls /root/ -alR
ls /opt/ -alR
# work inside include folder
cd /opt/hostedtoolcache/Python/$1*/x64/include;

echo "Working with python$1"

# Create symlink without "m" in name
[[ -f "python$1" ]] || ln -s python$1m python$1;

# copy libraries and includes to global paths
cp -r /opt/hostedtoolcache/Python/$1*/x64/lib/* /usr/local/lib/;
cp -r /opt/hostedtoolcache/Python/$1*/x64/include/py* /usr/include/;
ln -s /usr/include/$1m /usr/include/$1;

# Create symlink without "m" in name
[[ -f "/usr/local/lib/libpython$1.so.1.0" ]] ||  ln -s /usr/local/lib/libpython$1m.so.1.0 /usr/local/lib/libpython$1.so.1.0;
[[ -f "/usr/local/lib/libpython$1.so" ]] ||  ln -s /usr/local/lib/libpython$1m.so /usr/local/lib/libpython$1.so;
