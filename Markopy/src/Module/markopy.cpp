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
            .def("Train", &Markov::API::MarkovPasswords::Train, 
            "Train the model\n"
            "\n"
            ":param datasetFileName: Ifstream* to the dataset. If null, use class member\n"
            ":param delimiter: a character, same as the delimiter in dataset content\n"
            ":param threads: number of OS threads to spawn\n")
            .def("Generate", &Markov::API::MarkovPasswords::Generate, 
            "Generate passwords from a trained model.\n"
            ":param n: Ifstream* to the dataset. If null, use class member\n"
            ":param wordlistFileName: a character, same as the delimiter in dataset content\n"
            ":param minLen: number of OS threads to spawn\n"
            ":param maxLen: Ifstream* to the dataset. If null, use class member\n"
            ":param threads: a character, same as the delimiter in dataset content\n"
            ":param threads: number of OS threads to spawn\n")
            .def("Import", Import, "Import a model file.")
            .def("Export", Export, "Export a model to file.")
        ;
    };
};


