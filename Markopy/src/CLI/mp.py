""" @package markopy
 @file mp_cli.py
 @namespace Python::Markopy::MarkovPasswords
 @brief Command line class for MarkovPasswords
 @authors Ata Hakçıl
"""
from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader, ExtensionFileLoader
import os
ext = "so"
if os.name == 'nt':
    ext="pyd"
try:
    spec = spec_from_loader("markopy", ExtensionFileLoader("markopy", os.path.abspath(f"markopy.{ext}")))
    markopy = module_from_spec(spec)
    spec.loader.exec_module(markopy)
except ImportError as e:
    print(f"({__file__}) Working in development mode. Trying to load markopy.{ext} from ../../../out/")
    if(os.path.exists(f"../../../out/lib/markopy.{ext}")):
        spec = spec_from_loader("markopy", ExtensionFileLoader("markopy", os.path.abspath(f"../../../out/lib/markopy.{ext}")))
        markopy = module_from_spec(spec)
        spec.loader.exec_module(markopy)
    else:
        raise e

from base import BaseCLI,AbstractGenerationModelCLI, AbstractTrainingModelCLI



class MarkovPasswordsCLI(AbstractTrainingModelCLI):
    def __init__(self):
        super().__init__()
        self.model = markopy.MarkovPasswords()

    def _generate(self, wordlist):
        self.model.Generate(int(self.args.count), wordlist, int(self.args.min), int(self.args.max), int(self.args.threads))

if __name__ == "__main__":
    mp = MarkovPasswordsCLI()
    mp.parse()
    mp.process()