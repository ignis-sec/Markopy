FROM nvidia/cuda:11.0.3-devel-ubuntu20.04
MAINTAINER <hidden>

# Install add-apt-repository
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    rm -rf /var/lib/apt/lists/*

# Add deadsnake source for old python versions
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get -y update \
    && apt-get -y autoclean \
    && apt-get -y autoremove

# Install the C++ and python dependencies
RUN apt-get -y install \
        build-essential \
        cmake   \
        git \
        wget \
        python3.10-dev\
        python3.9-dev\
        python3.8-dev\
        python3.7-dev\
        python3.6-dev

# Download and install boost 1.76.0
RUN wget https://boostorg.jfrog.io/artifactory/main/release/1.76.0/source/boost_1_76_0.tar.gz -O /root/boost_1_76_0.tar.gz -q;

WORKDIR /root/

RUN tar -xf /root/boost_1_76_0.tar.gz; \
    rm /root/boost_1_76_0.tar.gz;

WORKDIR /root/boost_1_76_0

RUN ./bootstrap.sh; \
    ./b2 variant=release link=static threading=multi --with-program_options install;

RUN ./bootstrap.sh --with-python=$(which python3.6) --with-python-version=3.6; \
    ./b2 --with-python --buildid=3.6 install; \
    ./bootstrap.sh --with-python=$(which python3.7) --with-python-version=3.7; \
    ./b2 --with-python --buildid=3.7 install; \
    ./bootstrap.sh --with-python=$(which python3.8) --with-python-version=3.8; \
    ./b2 --with-python --buildid=3.8 install; \
    ./bootstrap.sh --with-python=$(which python3.9) --with-python-version=3.9; \
    ./b2 --with-python --buildid=3.9 install; \
    ./bootstrap.sh --with-python=$(which python3.10) --with-python-version=3.10; \
    ./b2 --with-python --buildid=3.10 install; 

RUN apt-get install -y qt5-default qtwebengine5-dev

WORKDIR /root

RUN rm -r /root/boost_1_76_0

RUN apt-get install zip