""" @package markopy
 @file cudammx_cli.py
 @namespace Python::Markopy::CUDA::ModelMatrix
 @brief Command line class for CudaModelMatrix
 @authors Ata Hakçıl
"""

import cudamarkopy
import os
import sys
try:
    from mmx_cli import ModelMatrixCLI
    from base_cli import BaseCLI,AbstractGenerationModelCLI, AbstractTrainingModelCLI
except ImportError as e:
    print("markopy_cli.py not found. Checking as if in project directory.")
    if(os.path.exists("../../../Markopy/src/CLI/markopy_cli.py")):
        sys.path.insert(1, '../../../Markopy/src/CLI/')
        from mmx_cli import ModelMatrixCLI
        from base_cli import BaseCLI,AbstractGenerationModelCLI, AbstractTrainingModelCLI
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