# Markopy


[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
![GitHub](https://img.shields.io/github/license/ignis-sec/Markopy?style=for-the-badge)


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">Markopy</h3>
  <div align="center">
   Generate wordlists with markov models.
    <br />
    <a href="https://markov.ignis.wtf">HTML documentation</a>
    ·
    <a href="https://markov.ignis.wtf/documentation.pdf">PDF documentation</a>
    ·
    <a href="https://github.com/ignis-sec/Markopy">Github Page</a>
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
          <li><a href="#possible-use-cases">Possible Use Cases</a></li>
          <li><a href="#getting-started">Getting Started</a></li>
          <li><a href="#releases">Releases</a></li>
        </ul>
      </li>
      <li>
        <a href="#building">Building</a>
        <ul>
          <li>
            <a href="#cmake-configuration">CMake configuration</a>
            <ul>
              <li><a href="#build-all">Build everything</a></li>
              <li><a href="#build-libs">Build libraries only</a></li>
              <li><a href="#build-cuda-libs">Build CUDA-accelerated libraries</a></li>
              <li><a href="#build-py-libs">Build python module & libraries</a></li>
              <li><a href="#build-cpy-libs">Build CUDA accelerated python module</a></li>
              <li><a href="#build-cpy-libs-and-gui">Build CUDA accelerated python module and the GUI</a></li>
            </ul>
          </li>
          <li><a href="#prequisites">Prequisites</a>
            <ul>
              <li>
                <a href="#general-prequisites">General Prequisites</a>
                <ul>
                  <li><a href="#general-prequisites-linux">Linux</a></li>
                  <li><a href="#general-prequisites-win">Windows</a></li>
                </ul>
              </li>
              <li><a href="#preq-mm">MarkovModel</a></li>
              <li><a href="#preq-ma">MarkovAPI</a></li>
              <li><a href="#preq-mac">MarkovAPICLI</a></li>
              <li><a href="#preq-mpy">Markopy</a></li>
              <li><a href="#preq-cma">CudaMarkovAPI</a></li>
              <li><a href="#preq-cmpy">CudaMarkopy</a></li>
              <li><a href="#preq-mpgui">MarkovPasswordsGUI</a></li>
            </ul>
          </li>
          <li>
            <a href="#installing-deps">Installing Dependencies</a>
            <ul>
              <li><a href="#installing-deps-win">Windows</a></li>
              <li><a href="#installing-deps-lin">Linux</a></li>
            </ul>
          </li>
        </ul>
      </li>
      <li><a href="#known-issues">Known Common Issues</a></li>
      <li><a href="#contributing">Contributing</a></li>
      <li><a href="#contact">Contact</a></li>
    </ol>
</div>

---

## About The Project
<div id="about-the-project"></div>

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
  - GPU-accelereted wrapper for CudaMarkovAPI


### Possible Use Cases
<div id="possible-use-cases"></div>

While main focus of the development has been towards random walk performance and password generation, underlying libraries could be used for other applications such as specific use cases of hidden markov models in bioinformatics and gene research.


### Getting Started
<div id="getting-started"></div>

If you'd just like to use the project without contributing, check out the releases page.
Latest minor release (0.8.x, 0.9.x) is even with main branch, and latest patch release (0.8.1, 0.8.2) is even with development branch.


### Releases
<div id="releases"></div>

Releases are maintained automatically via github actions. Each push to the main branch will trigger a minor version release, while each accepted pull request into the development branch will trigger a patch version release. 

Pull requests to the development branch will also trigger a draft release only visible to the maintainers.

Release files contain:
- libmarkov-{version}-{platform}.zip
  - Depending on the platform, contains the libmarkov.so or markov.lib from that version.
- libcudamarkov-{version}-{platform}.zip
  - Depending on the platform, contains the libcudamarkov.so or cudamarkov.lib from that version.
- markopy-{version}-{platform}-py{ver}.{extension}.zip
  - Depending on the paltform, contains markopy.so or markopy.pyd. CPython extensions are compiled for specific versions.
    If your python version is not supported by the releases, you can create an issue, or build it yourself using the python3.x-dev package.
- cudamarkopy-{version}-{platform}-py{ver}.{extension}.zip
  - Depending on the paltform, contains cudamarkopy.so or cudamarkopy.pyd. CPython extensions are compiled for specific versions.
    If your python version is not supported by the releases, you can create an issue, or build it yourself using the python3.x-dev package.
- models-{version}.zip
  - Contains the latest models with the release version.
    Contains base models (untrained), trained models, and language-specific models.

  
---
## Building
<div id="building"></div>
You can build the project using cmake with g++ & nvcc on linux, and msbuild & nvcc on windows.

## Prerequisites
<div id="prequisites"></div>

You can find a list of the dependencies below. If you have any missing, check out the <a href="#setting-up-deps">setting up prequisites</a> part.

### General prequisites
<div id="general-prequisites"></div>

To build the simple core of this project, you'll need:
#### Linux
- CMake, preferably one of the latest versions.
- CXX compiler, preferably g++ or clang++ (LLVM 3.9+).

#### Windows
- CMake, preferably one of the latest versions.
- CXX compiler, preferably msbuild(cl.exe) or clang++ (LLVM 3.9+). 
  Please note that mingw is not recommended as it is not officially supported by the nvcc.exe, and might not be linkable if you are building the CUDA components too.



### MarkovModel
<div id="preq-mm"></div>
This project does not have any extra dependencies, and it can be compiled with general dependencies without anything extra.

### MarkovAPI
<div id="preq-ma"></div>
This project does not have any extra dependencies, and it can be compiled with general dependencies without anything extra.

### MarkovAPICLI
<div id="preq-mac"></div>

- Boost.program_options (tested on 1.71.0-1.76.0)

### Markopy
<div id="preq-mac"></div>

- Boost.Python (tested on 1.71.0-1.76.0)
- Python development package (tested on python 36-39)

### CudaMarkovAPI
<div id="preq-cma"></div>

- CUDA toolkit (11.0+, c++17 support required)

### CudaMarkopy
<div id="preq-cmpy"></div>

- CUDA toolkit (11.0+, c++17 support required)
- Boost.Python (tested on 1.71.0-1.76.0)
- Python development package (tested on python 36-39)

### MarkovPasswordsGUI
<div id="preq-mpgui"></div>

- QT5 development environment. (qt5-qmake on ubuntu apt-get)
- QTWebEngine5 plugin. (qtwebengine5-dev on ubuntu apt-get)


### CMake Configuration
<div id="cmake-configuration"></div>

You can build this project with cmake. 

If you don't have prequisites for some of the projects set up, you can use the partial set up configuration to ignore those projects when setting the project up.

If you do not meet the prequisites, you'll have to partially set up the CMake file (you cant use --target to build some of the targets, because configuration phase will fail too).

Some examples for partially setting up and building the project are below.

#### Build everything
<div id="build-all"></div>

```
$ cmake . -DPYTHON_VER=38 && cmake --build .
```

This will build all the libraries and executables. Requires python-dev, CUDA, QT5, QT5-Webview

#### Build libraries only
<div id="build-libs"></div>

```
$ cmake . -DPARTIAL=1 -DB_LIBS=1 && cmake --build .
```

Only build basic libraries. Requires only CXX compiler.

#### Build CUDA-accelerated libraries
<div id="build-cuda-libs"></div>

```
$ cmake . -DPARTIAL=1 -DB_CUDA=1 && cmake --build .
```

Build libraries along with cuda accelerated ones.

#### Build python module & libraries
<div id="build-py-libs"></div>

```
$ cmake . -DPARTIAL=1 -DPYTHON_VER=39 && cmake --build .
```

Will build basic libraries and python modules.

#### Build CUDA accelerated python module
<div id="build-cpy-libs"></div>

```
$ cmake . -DPARTIAL=1 -DPYTHON_VER=39 -DB_CUDA=1 && cmake --build .
```

Will build cudamarkopy.

#### Build CUDA accelerated python module and the GUI
<div id="build-cpy-libs-and-gui"></div>

```
$ cmake . -DPARTIAL=1 -DPYTHON_VER=39 -DB_CUDA=1 -DB_GUI && cmake --build .
```

Combine methods


---

### Installing Dependencies
<div id="installing-deps"></div>

##### Windows
<div id="installing-deps-win"></div>

- QT: Install [QT For Windows](https://doc.qt.io/qt-5/windows.html)
- Boost (program_options and python): 
   - Download Boost from [its website](https://www.boost.org/users/download/). Prefer one of the tested versions, 1.71.0 to 1.76.0
   - Unzip the contents.
   - Launch "Visual Studio Developer Command Prompt" (If you don't have this, properly set up the %PATH% variable for cl.exe) 
   - Move to the boost installation directory. Bootstrap libraries with your python version:
    ```
    .\bootstrap.bat --with-python=$(which python3.6) --with-python-version=3.6;
    ```
   - Run `b2` to build the libraries.
    ```
    .\b2.exe --layout=system address-model=64 variant=release link=static runtime-link=shared threading=multi --with-program_options --with-python stage;
    ```

- Python: You can use the windows app store to download python runtime and libraries.


##### Linux
<div id="installing-deps-lin"></div>

- QT: Follow [this guide](https://wiki.qt.io/Install_Qt_5_on_Ubuntu) to install QT on Linux.
  Alternatively, on ubuntu you can `sudo apt-get install qt5-qmake qtwebengine5-dev`
- Boost (program options and python):
  - Download Boost from [its website](https://www.boost.org/users/download/). Prefer one of the tested versions, 1.71.0 to 1.76.0
  - Unzip the contents.
  - Move to the boost installation directory. Bootstrap libraries with your python version:
    ```
    ./bootstrap.sh --with-python=$(which python3.6) --with-python-version=3.6;
    ```
  - Run `b2` to build the libraries.
    ```
    ./b2 variant=release link=static threading=multi --with-program_options install;
    ./b2 --with-python --buildid=3.6 install;
    ```

- Boost (alternative)
  - Use a package manager to install boost
  ```
  sudo apt-get install libboost-all-dev
  ```

- Python:
  ```
  sudo apt-get install python3-dev
  ```

---


## Known Common issues
<div id="known-issues"></div>

### Linux
#### Markopy - Python.h - Not found
Make sure you have the development version of python package, which includes the required header files.
Check if header files exist: `/usr/include/python*` or `locate Python.h`.

If it doesn't, run `sudo apt-get install python3-dev`

#### Markopy/MarkovAPI - *.so not found, or other library related issues when building
Run `ls /usr/lib/x86_64-linux-gnu/ | grep boost` and check the shared object filenames. A common issue is that lboost is required but filenames are formatted as llibboost, or vice versa.

Do the same for python related library issues, run: `ls /usr/lib/x86_64-linux-gnu/ | grep python` to verify filename format is as required.

If not, you can modify the makefile, or create symlinks such as:
`ln -s /usr/lib/x86_64-linux-gnu/libboost_python38.so /usr/lib/x86_64-linux-gnu/boost_python38.so`

### Windows
#### Boost - Bootstrap.bat "ctype.h" not found
- Make sure you are working in the "Visual Studio Developer Command Prompt" terminal.
- Make sure you have Windows 10 SDK installed.
- From VS developer terminal, run echo %INCLUDE%. If result does not have the windows sdk folders, run the following before running bootstrap (change your sdk version instead of 10.0.19041.0):
```bat
set INCLUDE=%INCLUDE%;C:\Program Files (x86)\Windows Kits\NETFXSDK\4.8\include\um;C:\Program Files (x86)\Windows Kits\10\include\10.0.19041.0\ucrt;C:\Program Files (x86)\Windows Kits\10\include\10.0.19041.0\shared;C:\Program Files (x86)\Windows Kits\10\include\10.0.19041.0\um;C:\Program Files (x86)\Windows Kits\10\include\10.0.19041.0\winrt;C:\Program Files (x86)\Windows Kits\10\include\10.0.19041.0\cppwinrt

set LIB=%LIB%;C:\Program Files (x86)\Windows Kits\10\lib\10.0.19041.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\lib\10.0.19041.0\um\x64
```

#### Cannot open file "*.lib"
Make sure you have set the BOOST_ROOT environment variable correctly. Make sure you ran `b2` to build library files from boost sources.

#### Python.h not found
Make sure you have python installed, and make sure you set PYTHON_PATH environment variable.

---


<!-- CONTRIBUTING -->
## Contributing
<div id="contributing"></div>

Feel free to contribute. We welcome all the issues and pull requests.


<!-- CONTACT -->
## Contact
<div id="contact"></div>

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
