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
file = open('../../models/2gram.mdl', "wb")
#tie start nodes
for sym in alphabet:
	f.write(f"\x00,0,{sym}\n".encode())

#tie terminator nodes
for sym in alphabet:
	f.write(f"{sym},0,\xff\n".encode())

#tie internals
for src in alphabet:
	for target in alphabet:
		f.write(f"{src},0,{target}\n".encode())