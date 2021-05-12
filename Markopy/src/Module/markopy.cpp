#pragma once
#include "../../../MarkovPasswords/src/markovPasswords.h"


#include <Python.h>
#include <boost/python.hpp>

using namespace boost::python;

std::random_device rd;
std::default_random_engine generator(rd());
std::uniform_int_distribution<long long unsigned> distribution(0, 0xffffFFFF);

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
