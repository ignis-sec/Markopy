/** @file cudaMarkopy.cpp
 * @brief CPython wrapper for libcudamarkov utils. GPU
 * @authors Ata Hakçıl
 * 
 * @copydoc markopy.cpp
 * @copydoc cudaModelMatrix.cu
 */

#define BOOST_PYTHON_STATIC_LIB
#include <Python.h>
#include <boost/python.hpp>
#include "CudaMarkovAPI/src/cudaModelMatrix.h"

using namespace boost::python;

/**
 * @brief CPython module for Markov::API::CUDA objects
 */
namespace Markov::Markopy::CUDA{
    BOOST_PYTHON_MODULE(cudamarkopy)
    {
        bool (Markov::API::MarkovPasswords::*Export)(const char*) = &Markov::Model<char>::Export;
        void (Markov::API::CUDA::CUDAModelMatrix::*FastRandomWalk)(unsigned long int, const char*, int, int, bool, bool) = &Markov::API::CUDA::CUDAModelMatrix::FastRandomWalk;

        class_<Markov::API::CUDA::CUDAModelMatrix>("CUDAModelMatrix", init<>())
            
            .def(init<>())
            .def("Train", &Markov::API::ModelMatrix::Train)
            .def("Import", &Markov::API::ModelMatrix::Import, "Import a model file.")
            .def("Export", Export, "Export a model to file.")
            .def("ConstructMatrix",&Markov::API::ModelMatrix::ConstructMatrix)
            .def("DumpJSON",&Markov::API::ModelMatrix::DumpJSON)
            .def("FastRandomWalk", FastRandomWalk)
            ;
    };
};


