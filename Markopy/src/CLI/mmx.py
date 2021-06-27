""" @package markopy
 @file mmx_cli.py
 @namespace Python::Markopy::ModelMatrix
 @brief Command line class for ModelMatrix
 @authors Ata Hakçıl
"""

import markopy
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