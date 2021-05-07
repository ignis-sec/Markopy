#pragma once
#include "markovPasswords.h"


#include <Python.h>
#include <boost/python.hpp>
#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <boost/python/extract.hpp>



using namespace boost::python;
BOOST_PYTHON_MODULE(markopy)
{
    class_<MarkovPasswords>("MarkovPasswords")
        .def("train", &MarkovPasswords::Train)
        .def("generate", &MarkovPasswords::Generate)
    ;
};