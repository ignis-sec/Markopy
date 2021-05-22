#pragma once
#include "../../../MarkovPasswords/src/markovPasswords.h"

#define BOOST_PYTHON_STATIC_LIB
#include <Python.h>
#include <boost/python.hpp>

using namespace boost::python;


BOOST_PYTHON_MODULE(markopy)
{
    bool (MarkovPasswords::*Import)(const char*) = &Markov::Model<char>::Import;
    bool (MarkovPasswords::*Export)(const char*) = &Markov::Model<char>::Export;
    class_<MarkovPasswords>("MarkovPasswords", init<>())
        .def(init<>())
        .def("Train", &MarkovPasswords::Train)
        .def("Generate", &MarkovPasswords::Generate)
        .def("Import", Import)
        .def("Export", Export)
    ;
};


