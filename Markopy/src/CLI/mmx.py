""" @package markopy
 @file mmx_cli.py
 @namespace Python::Markopy::ModelMatrix
 @brief Command line class for ModelMatrix
 @authors Ata Hakçıl
"""

from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader, ExtensionFileLoader
import os
ext = "so"
if os.name == 'nt':
    ext="pyd"
try:
    spec = spec_from_loader("markopy", ExtensionFileLoader("markopy", f"markopy.{ext}"))
    markopy = module_from_spec(spec)
except ImportError as e:
    print(f"Working in development mode. Trying to load markopy.{ext} from ../../../out/")
    if(os.path.exists("../../../out/lib/markopy.so")):
        spec = spec_from_loader("markopy", ExtensionFileLoader("markopy", f"../../../out/lib/markopy.{ext}"))
        markopy = module_from_spec(spec)
    else:
        raise e
        
from base import BaseCLI, AbstractGenerationModelCLI
import os
import allogate as logging

class ModelMatrixCLI(AbstractGenerationModelCLI):
    def __init__(self):
        super().__init__()
        self.model = markopy.ModelMatrix()

    def add_arguments(self):
        super().add_arguments()
        self.parser.add_argument("-st", "--stdout", action="store_true", help="Stdout mode")
    
    def init_post_arguments(self):
        super().init_post_arguments()
        self.fileIO = not self.args.stdout
        
    def _generate(self, wordlist : str, ):
        self.model.FastRandomWalk(int(self.args.count), wordlist, int(self.args.min), int(self.args.max), int(self.args.threads), self.fileIO)

if __name__ == "__main__":
    mp = ModelMatrixCLI()
    mp.parse()
    mp.process()