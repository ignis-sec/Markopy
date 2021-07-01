#!/usr/bin/env python3


##
# @namespace Python.Markopy
# @brief wrapper scripts for Markopy
#

##
# @namespace Python
# @brief Python language scripts
#

##
# @file markopy.py
# @brief Entry point for markopy scripts.
#

from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader, ExtensionFileLoader
import os
import sys

ext = "so"
if os.name == 'nt':
    ext="pyd"


try:
    from base import BaseCLI
    from mp import MarkovPasswordsCLI
    from mmx import ModelMatrixCLI
    from evaluate import CorpusEvaluator, ModelEvaluator
    from importer import import_markopy
    markopy = import_markopy()


except ModuleNotFoundError as e:
    #print("Working in development mode. Trying to load markopy.py from ../../../Markopy/")
    if(os.path.exists("../../../Markopy/src/CLI/base.py")):
        sys.path.insert(1, '../../../Markopy/src/CLI/')
        from base import BaseCLI
        from mp import MarkovPasswordsCLI
        from mmx import ModelMatrixCLI
        from evaluate import CorpusEvaluator, ModelEvaluator
        from importer import import_markopy
        markopy = import_markopy()

    else:
        raise e 


from termcolor import colored
from abc import abstractmethod

class MarkopyCLI(BaseCLI):
    """!
        @brief Top level model selector for Markopy CLI.
        This class is used for injecting the -mt parameter to the CLI, and determining the model type depending on that.
        @belongsto Python::Markopy
        @extends Python::Markopy::BaseCLI
        @extends Python::Markopy::ModelMatrixCLI
        @extends Python::Markopy::MarkovPasswordsCLI
    """

    def __init__(self, add_help=False):
        """! 
        @brief default constructor
        """

        BaseCLI.__init__(self,add_help)
        self.args = None
        self.parser.epilog = f"""
        {colored("Sample runs:", "yellow")}
        {__file__.split("/")[-1]} -mt MP generate trained.mdl -n 500 -w output.txt
            Import trained.mdl, and generate 500 lines to output.txt

        {__file__.split("/")[-1]} -mt MMX generate trained.mdl -n 500 -w output.txt
            Import trained.mdl, and generate 500 lines to output.txt
        """

    @abstractmethod
    def add_arguments(self):
        """! 
        @brief add -mt/--model_type constructor
        """
        self.parser.add_argument("-mt", "--model_type", default="_MMX", help="Model type to use. Accepted values: MP, MMX")
        self.parser.add_argument("-h", "--help", action="store_true", help="Model type to use. Accepted values: MP, MMX")
        self.parser.add_argument("-ev", "--evaluate", help="Evaluate a models integrity")
        self.parser.add_argument("-evt", "--evaluate_type", help="Evaluation type, model or corpus")
        self.parser.print_help = self.help

    @abstractmethod
    def help(self):
        """! 
        @brief overload help function to print submodel helps
        """
        self.parser.print_help = self.stub
        self.args = self.parser.parse_known_args()[0]
        if(self.args.model_type!="_MMX"):
            if(self.args.model_type=="MP"):
                mp = MarkovPasswordsCLI()
                mp.add_arguments()
                mp.parser.print_help()
            elif(self.args.model_type=="MMX"):
                mp = ModelMatrixCLI()
                mp.add_arguments()
                mp.parser.print_help()
        else:
            print(colored("Model Mode selection choices:", "green"))
            self.print_help()
            print(colored("Following are applicable for -mt MP mode:", "green"))
            mp = MarkovPasswordsCLI()
            mp.add_arguments()
            mp.parser.print_help()
            print(colored("Following are applicable for -mt MMX mode:", "green"))
            mp = ModelMatrixCLI()
            mp.add_arguments()
            mp.parser.print_help()

        exit()


    @abstractmethod
    def parse(self):
        "! @brief overload parse function to parse for submodels"
        
        self.add_arguments()
        self.parse_arguments()
        self.init_post_arguments()
        if(self.args.evaluate): 
            self.evaluate(self.args.evaluate)
            exit()
        if(self.args.model_type == "MP"):
            self.cli = MarkovPasswordsCLI()
        elif(self.args.model_type == "MMX" or self.args.model_type == "_MMX"):
            self.cli = ModelMatrixCLI()
        else:
            self.parse_fail()

        if(self.args.help): return self.help()
        self.cli.parse()
    
    @abstractmethod
    def init_post_arguments(self):
        pass

    @abstractmethod
    def parse_fail(self):
        "! @brief failed to parse model type"
        print("Unrecognized model type.")
        exit()

    def process(self):
        "! @brief pass the process request to selected submodel"
        return self.cli.process()

    def stub(self):
        "! @brief stub function to hack help requests"
        return
        
    
    def evaluate(self,filename : str):
        if(not self.args.evaluate_type):
            if(filename.endswith(".mdl")):
                ModelEvaluator(filename).evaluate()
            else:
                CorpusEvaluator(filename).evaluate()
        else:
            if(self.args.evaluate_type == "model"):
                ModelEvaluator(filename).evaluate()
            else:
                CorpusEvaluator(filename).evaluate()
    
    def init_post_arguments(sel):
        pass

if __name__ == "__main__":
    mp = MarkopyCLI()
    mp.parse()
    mp.process()