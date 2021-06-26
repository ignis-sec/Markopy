""" @package cudamarkopy
 @file cudamarkopy_cli.py
 @namespace Python::CUDA::Markopy
 @brief Command line 
 @authors Ata Hakçıl
"""

import sys
import os
import cudamarkopy
from  cudammx_cli import CudaModelMatrixCLI
from mmx_cli import ModelMatrixCLI
from mp_cli import MarkovPasswordsCLI
from termcolor import colored

try:
    import markopy_cli
except ImportError as e:
    print("markopy_cli.py not found. Checking as if in project directory.")
    if(os.path.exists("../../../Markopy/src/CLI/markopy_cli.py")):
        sys.path.insert(1, '../../../Markopy/src/CLI/')
        import markopy_cli
    else:
        raise e


class CudaMarkopyCLI(markopy_cli.MarkopyCLI):
    def __init__(self) -> None:
        super().__init__(add_help=False)
        self.parser.epilog+=f"""
        {__file__.split("/")[-1]} -mt CUDA generate trained.mdl -n 500 -w output.txt
            Import trained.mdl, and generate 500 lines to output.txt
        """

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
            elif(self.args.model_type == "CUDA"):
                mp = CudaModelMatrixCLI()
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
            print(colored("Following are applicable for -mt CUDA mode:", "green"))
            mp = CudaModelMatrixCLI()
            mp.add_arguments()
            mp.parser.print_help()
        exit()

    def parse_fail(self):
        if(self.args.model_type == "CUDA"):
            self.cli = CudaModelMatrixCLI()
        else:
            super().parse_fail()
        

if __name__ == "__main__":
    mp = CudaMarkopyCLI()
    #mp = markopy_cli.MarkopyCLI()
    mp.parse()
    mp.process()