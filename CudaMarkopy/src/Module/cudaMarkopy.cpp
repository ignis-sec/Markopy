#pragma once

#define BOOST_PYTHON_STATIC_LIB
#include <Python.h>
#include <boost/python.hpp>
#include <MarkovPasswords/src/modelMatrix.h>

using namespace boost::python;

namespace Markov::Markopy{
    BOOST_PYTHON_MODULE(markopy)
    {


        class_<Markov::API::CUDA::CUDAModelMatrix>("CUDAModelMatrix", init<>())
            
            .def(init<>())
            .def("Train", &Markov::API::ModelMatrix::Train)
            .def("Import", &Markov::API::ModelMatrix::Import, "Import a model file.")
            .def("Export", Export, "Export a model to file.")
            .def("ConstructMatrix",&Markov::API::ModelMatrix::ConstructMatrix)
            .def("DumpJSON",&Markov::API::ModelMatrix::DumpJSON)
            .def("FastRandomWalk",&Markov::API::CUDAModelMatrix::FastRandomWalk)
            ;
    };
};


