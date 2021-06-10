#!/usr/bin/python3
""" 
  python script for generating a 2gram model
"""

import string
import re

## @brief password alphabet
alphabet = string.printable
alphabet = re.sub('\s', '', alphabet)
print(f"alphabet={alphabet}")
#exit()

## @brief output file handle
f = open('../../models/2gram.mdl', "wb")
#tie start nodes
for sym in alphabet:
	f.write(b"\x00,1," + bytes(sym, encoding='ascii') + b"\n")

#tie terminator nodes
for sym in alphabet:
	f.write(bytes(sym, encoding='ascii')+ b",1,\xff\n")

#tie internals
for src in alphabet:
	for target in alphabet:
		f.write(bytes(src, encoding='ascii') + b",1," + bytes(target, encoding='ascii') + b"\n")