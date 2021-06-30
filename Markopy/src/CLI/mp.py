#!/usr/bin/python3

##
# @file mp.py
# @brief CLI wrapper for MarkovPasswords
#

from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader, ExtensionFileLoader
import os
from mm import MarkovModel
from importer import import_markopy
markopy = import_markopy()

from base import BaseCLI,AbstractGenerationModelCLI, AbstractTrainingModelCLI

class MarkovPasswordsCLI(AbstractTrainingModelCLI,MarkovModel):
    """!
        @brief Extension of Python.Markopy.Base.BaseCLI for Markov::API::MarkovPasswords
        @belongsto Python::Markopy
        @extends Python::Markopy::MarkovModel
        @extends Python::Markopy::AbstractTrainingModelCLI

        adds -st/--stdout arguement to the command line.
    """
    def __init__(self, add_help:bool=True):
        "! @brief initialize model with Markov::API::MarkovPasswords"
        super().__init__(add_help)
        self.model = markopy.MarkovPasswords()

    def _generate(self, wordlist):
        "! @brief map generation function to Markov::API::MarkovPasswords::Generate"
        self.model.Generate(int(self.args.count), wordlist, int(self.args.min), int(self.args.max), int(self.args.threads))

if __name__ == "__main__":
    mp = MarkovPasswordsCLI()
    mp.parse()
    mp.process()