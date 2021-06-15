#pragma once
#include "../../../MarkovPasswords/src/markovPasswords.h"

#define BOOST_PYTHON_STATIC_LIB
#include <Python.h>
#include <boost/python.hpp>

using namespace boost::python;

namespace Markov::Markopy{
    BOOST_PYTHON_MODULE(markopy)
    {
        bool (Markov::API::MarkovPasswords::*Import)(const char*) = &Markov::Model<char>::Import;
        bool (Markov::API::MarkovPasswords::*Export)(const char*) = &Markov::Model<char>::Export;
        class_<Markov::API::MarkovPasswords>("MarkovPasswords", init<>())
            .def(init<>())
            .def("Train", &Markov::API::MarkovPasswords::Train)
            .def("Generate", &Markov::API::MarkovPasswords::Generate)
            .def("Import", Import)
            .def("Export", Export)
        ;
    };
};


