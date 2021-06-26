""" @package markopy
 @file mp_cli.py
 @namespace Python::Markopy::MarkovPasswords
 @brief Command line class for MarkovPasswords
 @authors Ata Hakçıl
"""

import markopy
from base_cli import BaseCLI,AbstractGenerationModelCLI, AbstractTrainingModelCLI



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