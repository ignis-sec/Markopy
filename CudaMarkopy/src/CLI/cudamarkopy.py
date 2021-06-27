""" @package cudamarkopy
 @file cudamarkopy_cli.py
 @namespace Python::CUDA::Markopy
 @brief Command line 
 @authors Ata Hakçıl
"""

import sys
import os
from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader, ExtensionFileLoader
import inspect 
try:
    spec = spec_from_loader("markopy", SourceFileLoader("markopy", "markopy.py"))
    markopy = module_from_spec(spec)
    spec.loader.exec_module(markopy)
except (ModuleNotFoundError,FileNotFoundError) as e:
    if(os.path.exists("../../../Markopy/src/CLI/markopy.py")):
        spec = spec_from_loader("markopy", SourceFileLoader("markopy", "../../../Markopy/src/CLI/markopy.py"))
        markopy = module_from_spec(spec)
        spec.loader.exec_module(markopy)

try:
    from cudammx import CudaModelMatrixCLI
    from mmx import ModelMatrixCLI
    from mp import MarkovPasswordsCLI

except ModuleNotFoundError as e:
    print("Working in development mode. Trying to load markopy.py from ../../../Markopy/")
    if(os.path.exists("../../../Markopy/src/CLI/cudammx.py")):
        sys.path.insert(1, '../../../Markopy/src/CLI/')
        from cudammx import CudaModelMatrixCLI
        from mmx import ModelMatrixCLI
        from mp import MarkovPasswordsCLI
    else:
        raise e 


#print(markopy)
#print(inspect.getmembers(markopy))
#import code
#code.interact(local=locals())

from termcolor import colored


class CudaMarkopyCLI(markopy.MarkopyCLI):
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