""" @package markopy
 @file cudammx_cli.py
 @namespace Python::Markopy::CUDA::ModelMatrix
 @brief Command line class for CudaModelMatrix
 @authors Ata Hakçıl
"""

from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader, ExtensionFileLoader
import os
import sys

ext = "so"
if os.name == 'nt':
    ext="pyd"
try:
    spec = spec_from_loader("cudamarkopy", ExtensionFileLoader("cudamarkopy", f"cudamarkopy.{ext}"))
    cudamarkopy = module_from_spec(spec)
except ImportError as e:
    print("Working in development mode. Trying to load cudamarkopy.so from ../../../out/")
    if(os.path.exists(f"../../../out/lib/cudamarkopy.{ext}")):
        spec = spec_from_loader("cudamarkopy", ExtensionFileLoader("cudamarkopy", f"../../../out/lib/cudamarkopy.{ext}"))
        cudamarkopy = module_from_spec(spec)
    else:
        raise e



try:
    spec = spec_from_loader("markopy", SourceFileLoader("markopy", "markopy.py"))
    markopy = module_from_spec(spec)

    from mmx import ModelMatrixCLI
    from base import BaseCLI,AbstractGenerationModelCLI, AbstractTrainingModelCLI

except ImportError as e:
    print("Working in development mode. Trying to load from ../../../out/")
    if(os.path.exists("../../../Markopy/src/CLI/markopy.py")):
        spec = spec_from_loader("markopy", SourceFileLoader("markopy", "../../../Markopy/src/CLI/markopy.py"))
        markopy = module_from_spec(spec)
        sys.path.insert(1, '../../../Markopy/src/CLI/')

        from mmx import ModelMatrixCLI
        from base import BaseCLI,AbstractGenerationModelCLI, AbstractTrainingModelCLI
    else:
        raise e

import os
import allogate as logging

class CudaModelMatrixCLI(ModelMatrixCLI,AbstractGenerationModelCLI):
    def __init__(self):
        super().__init__()
        self.model = cudamarkopy.CUDAModelMatrix()

    def add_arguments(self):
        super().add_arguments()
        self.parser.add_argument("-if", "--infinite", action="store_true", help="Infinite generation mode")
        


    def init_post_arguments(self):
        super().init_post_arguments()
        self.bInfinite = self.args.infinite
        
    def _generate(self, wordlist : str ):
        self.model.FastRandomWalk(int(self.args.count), wordlist, int(self.args.min), int(self.args.max), self.fileIO, self.bInfinite)

if __name__ == "__main__":
    mp = CudaModelMatrixCLI()
    mp.parse()
    mp.process()