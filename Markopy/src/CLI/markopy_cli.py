""" @package markopy
 @file markopy_cli.py
 @namespace Python::Markopy::ModelMatrix
 @brief Command line class for ModelMatrix
 @authors Ata Hakçıl
"""

import markopy
from base_cli import BaseCLI
from mp_cli import MarkovPasswordsCLI
from mmx_cli import ModelMatrixCLI
from termcolor import colored
from abc import abstractmethod

class MarkopyCLI(BaseCLI):
    def __init__(self, add_help=False):
        super().__init__(add_help)
        self.args = None
        self.parser.epilog = f"""
        {colored("Sample runs:", "yellow")}
        {__file__.split("/")[-1]} -mt MP generate trained.mdl -n 500 -w output.txt
            Import trained.mdl, and generate 500 lines to output.txt

        {__file__.split("/")[-1]} -mt MMX generate trained.mdl -n 500 -w output.txt
            Import trained.mdl, and generate 500 lines to output.txt
        """

    def add_arguments(self):
        self.parser.add_argument("-mt", "--model_type", default="_MMX", help="Model type to use. Accepted values: MP, MMX")
        self.parser.add_argument("-h", "--help", action="store_true", help="Model type to use. Accepted values: MP, MMX")
        self.parser.print_help = self.help

    @abstractmethod
    def help(self):
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
        self.add_arguments()
        self.parse_arguments()
        self.init_post_arguments()
        if(self.args.model_type == "MP"):
            self.cli = MarkovPasswordsCLI()
        elif(self.args.model_type == "MMX" or self.args.model_type == "_MMX"):
            self.cli = ModelMatrixCLI()
        else:
            self.parse_fail()

        if(self.args.help): return self.help()
        self.cli.parse()
    
    @abstractmethod
    def parse_fail(self):
        print("Unrecognized model type.")
        exit()

    def process(self):
        return self.cli.process()

    def stub(self):
        return
        

if __name__ == "__main__":
    mp = MarkopyCLI()
    mp.parse()
    mp.process()