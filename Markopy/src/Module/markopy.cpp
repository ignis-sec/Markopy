/** @file markopy.cpp
 * @brief CPython wrapper for libmarkov utils.
 * @authors Ata Hakçıl, Celal Sahir Çetiner
 * 
 * This file is a wrapper for libmarkov utilities, exposing:
 * - MarkovPasswords
 *   - Import
 *   - Export
 *   - Train
 *   - Generate
 * - ModelMatrix
 *   - Import
 *   - Export 
 *   - Train
 *   - ConstructMatrix
 *   - DumpJSON
 *   - FastRandomWalk
 * 
 * @copydoc Markov::API::MarkovPasswords
 * @copydoc Markov::API::ModelMatrix
 * 
*/

#define BOOST_ALL_STATIC_LIB 1
#define BOOST_PYTHON_STATIC_LIB 1
#include <Python.h>
#include <boost/python.hpp>
#include <MarkovAPI/src/modelMatrix.h>


using namespace boost::python;

/**
 * @brief CPython module for Markov::API objects
 */
namespace Markov::Markopy{
    BOOST_PYTHON_MODULE(markopy)
    {
        bool (Markov::API::MarkovPasswords::*Import)(const char*) = &Markov::Model<char>::Import;
        bool (Markov::API::MarkovPasswords::*Export)(const char*) = &Markov::Model<char>::Export;
        class_<Markov::API::MarkovPasswords>("MarkovPasswords", init<>())
            .def(init<>())
            .def("Train", &Markov::API::MarkovPasswords::Train)
            .def("Generate", &Markov::API::MarkovPasswords::Generate)
            .def("Import", Import, "Import a model file.")
            .def("Export", Export, "Export a model to file.")
        ;

        int (Markov::API::ModelMatrix::*FastRandomWalk)(unsigned long int, const char*, int, int, int, bool)
            = &Markov::API::ModelMatrix::FastRandomWalk;
        class_<Markov::API::ModelMatrix>("ModelMatrix", init<>())
            
            .def(init<>())
            .def("Train", &Markov::API::ModelMatrix::Train)
            .def("Import", &Markov::API::ModelMatrix::Import, "Import a model file.")
            .def("Export", Export, "Export a model to file.")
            .def("ConstructMatrix",&Markov::API::ModelMatrix::ConstructMatrix)
            .def("DumpJSON",&Markov::API::ModelMatrix::DumpJSON)
            .def("FastRandomWalk",FastRandomWalk)
            ;
    };
};


