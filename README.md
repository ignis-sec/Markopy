# Markov Passwords


[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
![GitHub](https://img.shields.io/github/license/ignis-sec/Markopy?style=for-the-badge)


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">Markov Passwords</h3>
  <div align="center">
   Generate wordlists with markov models.
    <br />
    <a href="https://github.com/ignis-sec/Markopy/wiki">Wiki</a>
    ·
    <a href="https://markov.ignis.wtf">Complete documentation</a>
    ·
    <a href="https://github.com/ignis-sec/Markopy/issues">Report Bug</a>
    ·
    <a href="https://github.com/ignis-sec/Markopy/pulls">Add a Bug</a>
  </div>
</div>



<div>
    <h4>Table of Contents</h4>
    <ol>
      <li>
        <a href="#about-the-project">About The Project</a>
        <ul>
          <li><a href="#built-with">Built With</a></li>
        </ul>
      </li>
      <li>
        <a href="#getting-started">Getting Started</a>
        <ul>
          <li><a href="#prerequisites">Prerequisites</a></li>
          <li><a href="#installation">Installation</a></li>
        </ul>
      </li>
      <li><a href="#contributing">Contributing</a></li>
      <li><a href="#contact">Contact</a></li>
    </ol>
</div>

---

## About The Project

This projects primary goal is to create a comfortable development environment for working with Markov Models,
as well as creating an end product which can be used for generating password wordlists using Markov Models.

This project contains following sub-projects:

- MarkovModel
  - A versatile header-only template library for basic Markov Model structure.
- MarkovAPI
  - A static/dynamic library built on MarkovModel, specialized to generate single-word lines.
- MarkovAPICLI
  - A command line interface built on top of MarkovAPI
- Markopy
  - A CPython extension wrapper for MarkovAPI, along with its own command line interface.
- MarkovPasswordsGUI
  - A graphical user interface for MarkovAPI
- CudaMarkovAPI
  - GPU-accelerated wrapper for MarkovAPI
- CudaMarkopy
  - GPU-accelereted wrağğer for CudaMarkovAPI


## Possible use cases

While main focus of the development has been towards random walk performance

### Built With

Each one of the 


## Getting Started

If you'd just like to use the project without contributing, check out the releases page. If you want to build, check out wiki for building the project.

---

### Prerequisites

##### MarkovModel
- Make for linux, Visual Studio/MSBuild for Windows.

##### MarkovPasswords
- Boost.ProgramOptions (tested on 1.76.0)

##### Markopy
- Boost.Python (tested on 1.76.0)
- Python development package (tested on python 3.8)

##### MarkovPasswordsGUI
- QT development environment.

---

### Installing Dependencies
##### Windows
- QT: Install [QT For Windows](https://doc.qt.io/qt-5/windows.html)
- Boost: 
   - Download Boost from [its website](https://www.boost.org/users/download/)
   - Unzip the contents.
   - Launch "Visual Studio Developer Command Prompt"
   - Move to the boost installation directory. Run `bootstrap.bat`
   - Run `b2`.
- Python: You can use the windows app store to download python runtime and libraries.


##### Linux
- QT: Follow [this guide](https://wiki.qt.io/Install_Qt_5_on_Ubuntu) to install QT on Linux.
- Boost: run `sudo apt-get install libboost-all-dev`
- Python: run `sudo apt-get install python3`

---

### Installation

See the Wiki Page

---

### Building
Building process can be fairly complicated depending on the environment.

## Linux
If you've set up the dependencies, you can just build the project with make.
List of directives is below. 

```makefile
.PHONY: all
all: model mp

model: $(INCLUDE)/$(MM_LIB)

mp: $(BIN)/$(MP_EXEC)

markopy: $(BIN)/$(MPY_SO)

.PHONY: clean
clean:
	$(RM) -r $(BIN)/*

```

## Windows
Set up correct environment variables for %BOOST_ROOT% (folder containing boost, libs, stage, tools) and %PYTHON_PATH% (folder containing include, lib, libs, Tools, python.exe/python3.exe).

If you've set up the dependencies and environment variables correctly, you can open the solution with Visual Studio and build with that.

---

# Known Common issues
## Linux
### Markopy - Python.h - Not found
Make sure you have the development version of python package, which includes the required header files.
Check if header files exist: `/usr/include/python*` 

If it doesn't, run `sudo apt-get install python3-dev`

### Markopy/MarkovPasswords - *.so not found, or other library related issues when building
Run `ls /usr/lib/x86_64-linux-gnu/ | grep boost` and check the shared object filenames. A common issue is that lboost is required but filenames are formatted as llibboost, or vice versa.

Do the same for python related library issues, run: `ls /usr/lib/x86_64-linux-gnu/ | grep python` to verify filename format is as required.

If not, you can modify the makefile, or create symlinks such as:
`ln -s /usr/lib/x86_64-linux-gnu/libboost_python38.so /usr/lib/x86_64-linux-gnu/boost_python38.so`

## Windows
### Boost - Bootstrap.bat "ctype.h" not found
- Make sure you are working in the "Visual Studio Developer Command Prompt" terminal.
- Make sure you have Windows 10 SDK installed.
- From VS developer terminal, run echo %INCLUDE%. If result does not have the windows sdk folders, run the following before running bootstrap (change your sdk version instead of 10.0.19041.0):
```bat
set INCLUDE=%INCLUDE%;C:\Program Files (x86)\Windows Kits\NETFXSDK\4.8\include\um;C:\Program Files (x86)\Windows Kits\10\include\10.0.19041.0\ucrt;C:\Program Files (x86)\Windows Kits\10\include\10.0.19041.0\shared;C:\Program Files (x86)\Windows Kits\10\include\10.0.19041.0\um;C:\Program Files (x86)\Windows Kits\10\include\10.0.19041.0\winrt;C:\Program Files (x86)\Windows Kits\10\include\10.0.19041.0\cppwinrt

set LIB=%LIB%;C:\Program Files (x86)\Windows Kits\10\lib\10.0.19041.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\lib\10.0.19041.0\um\x64
```

### Cannot open file "*.lib"
Make sure you have set the BOOST_ROOT environment variable correctly. Make sure you ran `b2` to build library files from boost sources.

### Python.h not found
Make sure you have python installed, and make sure you set PYTHON_PATH environment variable.

---

### Simplified Theory
##### What is a markov model

Below, is the example Markov Model which can generate strings with the alphabet "a,b,c"

![](https://raw.githubusercontent.com/ignis-sec/Markopy/main/docs/images/empty_model.png)


##### Iteration 1

Below is a demonstration of how training will be done. For this example, we are going to adjust the model with string "ab", and our occurrence will be "3"
From MarkovPasswords, inside the train function, Model::adjust is called with "ab" and "3" parameters.

Now, Model::adjust will iteratively adjust the edge weights accordingly. It starts by adjusting weight between start and "a" node. This is done by calling Edge::adjust of the edge between the nodes.

![](https://raw.githubusercontent.com/ignis-sec/Markopy/main/docs/images/model_1.png)

After adjustment, ajust function iterates to the next character, "b", and does the same thing.

![](https://raw.githubusercontent.com/ignis-sec/Markopy/main/docs/images/model_2.png)

As this string is finished, it will adjust the final weight, b->"end"

![](https://raw.githubusercontent.com/ignis-sec/Markopy/main/docs/images/model_3.png)

##### Iteration 2

This time, same procedure will be applied for "bacb" string, with occurrence value of 12.


![](https://raw.githubusercontent.com/ignis-sec/Markopy/main/docs/images/model_21.png)

![](https://raw.githubusercontent.com/ignis-sec/Markopy/main/docs/images/model_22.png)

![](https://raw.githubusercontent.com/ignis-sec/Markopy/main/docs/images/model_23.png)

![](https://raw.githubusercontent.com/ignis-sec/Markopy/main/docs/images/model_24.png)

![](https://raw.githubusercontent.com/ignis-sec/Markopy/main/docs/images/model_25.png)



##### Iteration 38271

As the model is trained, hidden linguistical patterns start to appear, and our model looks like this
![](https://raw.githubusercontent.com/ignis-sec/Markopy/main/docs/images/model_30.png)

With our dataset, without doing any kind of linugistic analysis ourselves, our Markov Model has highlighted that strings are more likely to start with a, b tends to follow a, and a is likely to be repeated in the string.
![](https://raw.githubusercontent.com/ignis-sec/Markopy/main/docs/images/model_31.png)

---


<!-- CONTRIBUTING -->
## Contributing
Feel free to contribute.


<!-- CONTACT -->
## Contact
Twitter - [@ahakcil](https://twitter.com/ahakcil)




[contributors-shield]: https://img.shields.io/github/contributors/ignis-sec/Markopy.svg?style=for-the-badge
[contributors-url]: https://github.com/ignis-sec/Markopy/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/ignis-sec/Markopy.svg?style=for-the-badge
[forks-url]: https://github.com/ignis-sec/Markopy/network/members
[stars-shield]: https://img.shields.io/github/stars/ignis-sec/Markopy.svg?style=for-the-badge
[stars-url]: https://github.com/ignis-sec/Markopy/stargazers
[issues-shield]: https://img.shields.io/github/issues/ignis-sec/Markopy.svg?style=for-the-badge
[issues-url]: https://github.com/ignis-sec/Markopy/issues
[license-shield]: https://img.shields.io/github/license/ignis-sec/Markopy.svg?style=for-the-badge
[license-url]: https://github.com/ignis-sec/Markopy/LICENSE
