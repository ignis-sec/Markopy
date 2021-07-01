#!/usr/bin/python3

##
# @file base.py
# @brief base command line interface for python
#

import argparse
import allogate as logging
import os
from abc import abstractmethod
from termcolor import colored
from mm import MarkovModel


class BaseCLI():
    """! @brief Base CLI class to handle user interactions        
         @belongsto Python::Markopy
    """
    def __init__(self, add_help : bool=True):
        """!
        @brief initialize base CLI
        @param add_help decide to overload the help function or not
        """
        self.parser = argparse.ArgumentParser(description="Python wrapper for MarkovPasswords.",
        epilog=f"""{colored("Sample runs:", "yellow")}
        {__file__.split("/")[-1]} train untrained.mdl -d dataset.dat -s "\\t" -o trained.mdl
            Import untrained.mdl, train it with dataset.dat which has tab delimited data, output resulting model to trained.mdl\n

        {__file__.split("/")[-1]} generate trained.mdl -n 500 -w output.txt
            Import trained.mdl, and generate 500 lines to output.txt

        {__file__.split("/")[-1]} combine untrained.mdl -d dataset.dat -s "\\t" -n 500 -w output.txt
            Train and immediately generate 500 lines to output.txt. Do not export trained model.

        {__file__.split("/")[-1]} combine untrained.mdl -d dataset.dat -s "\\t" -n 500 -w output.txt -o trained.mdl
            Train and immediately generate 500 lines to output.txt. Export trained model.
        """, add_help=add_help, formatter_class=argparse.RawTextHelpFormatter)
        self.print_help = self.parser.print_help
        self.model = MarkovModel()
     
    @abstractmethod
    def add_arguments(self):
        "! @brief Add command line arguements to the parser"
        self.parser.add_argument("mode",                             help="Process mode. Either 'Train', 'Generate', or 'Combine'.")
        self.parser.add_argument("-t", "--threads",default=10,       help="Number of lines to generate. Ignored in training mode.")
        self.parser.add_argument("-v", "--verbosity",action="count", help="Output verbosity.")
        self.parser.add_argument("-b", "--bulk",action="store_true", help="Bulk generate or bulk train every corpus/model in the folder.")

    @abstractmethod
    def help(self):
        "! @brief Handle help strings. Defaults to argparse's help"
        self.print_help()

    def parse(self):
        "! @brief add, parse and hook arguements"
        self.add_arguments()
        self.parse_arguments()
        self.init_post_arguments()

    @abstractmethod
    def init_post_arguments(self):
        "! @brief set up stuff that is collected from command line arguements"
        logging.VERBOSITY = 0
        try:
            if self.args.verbosity:
                logging.VERBOSITY = self.args.verbosity
                logging.pprint(f"Verbosity set to {self.args.verbosity}.", 2)
        except:
            pass
    
    @abstractmethod
    def parse_arguments(self):
        "! @brief trigger parser"
        self.args = self.parser.parse_known_args()[0]

    def import_model(self, filename : str):
        """! 
        @brief Import a model file
        @param filename filename to import
        """
        logging.pprint("Importing model file.", 1)

        if not self.check_import_path(filename):
            logging.pprint(f"Model file at {filename} not found. Check the file path, or working directory")
            return False

        self.model.Import(filename)
        logging.pprint("Model imported successfully.", 2)
        return True



    def train(self, dataset : str, seperator : str, output : str, output_forced : bool=False, bulk : bool=False):
        """! 
            @brief Train a model via CLI parameters 
            @param model Model instance
            @param dataset filename for the dataset
            @param seperator seperator used with the dataset
            @param output output filename
            @param output_forced force overwrite
            @param bulk marks bulk operation with directories
        """
        logging.pprint("Training.")

        if not (dataset and seperator and (output or not output_forced)):
            logging.pprint(f"Training mode requires -d/--dataset{', -o/--output' if output_forced else''} and -s/--seperator parameters. Exiting.")
            return False

        if not bulk and not self.check_corpus_path(dataset):
            logging.pprint(f"{dataset} doesn't exists. Check the file path, or working directory")
            return False

        if not self.check_export_path(output):
            logging.pprint(f"Cannot create output at {output}")
            return False

        if(seperator == '\\t'):
            logging.pprint("Escaping seperator.", 3)
            seperator = '\t'
        
        if(len(seperator)!=1):
            logging.pprint(f'Delimiter must be a single character, and "{seperator}" is not accepted.')
            exit(4)

        logging.pprint(f'Starting training.', 3)
        self.model.Train(dataset,seperator, int(self.args.threads))
        logging.pprint(f'Training completed.', 2)

        if(output):
            logging.pprint(f'Exporting model to {output}', 2)
            self.export(output)
        else:
            logging.pprint(f'Model will not be exported.', 1)

        return True

    def export(self, filename : str):
        """! 
        @brief Export model to a file
        @param filename filename to export to
        """
        self.model.Export(filename)

    def generate(self, wordlist : str, bulk : bool=False):
        """! 
            @brief Generate strings from the model
            @param model: model instance
            @param wordlist wordlist filename
            @param bulk marks bulk operation with directories
        """
        if not (wordlist or self.args.count):
            logging.pprint("Generation mode requires -w/--wordlist and -n/--count parameters. Exiting.")
            return False
    
        if(bulk and os.path.isfile(wordlist)):
            logging.pprint(f"{wordlist} exists and will be overwritten.", 1)
        self._generate(wordlist)

    @abstractmethod
    def _generate(self, wordlist : str):
        """!
        @brief wrapper for generate function. This can be overloaded by other models
        @param wordlist filename to generate to
        """
        self.model.Generate(int(self.args.count), wordlist, int(self.args.min), int(self.args.max), int(self.args.threads))

    @staticmethod
    def check_import_path(filename : str):
        """!
        @brief check import path for validity
        @param filename filename to check
        """
        
        if(not os.path.isfile(filename)):
            return False
        else:
            return True

    @staticmethod
    def check_corpus_path(filename : str):
        """!
        @brief check import path for validity
        @param filename filename to check
        """

        if(not os.path.isfile(filename)):
            return False
        return True

    @staticmethod
    def check_export_path(filename : str):
        """!
        @brief check import path for validity
        @param filename filename to check
        """

        if(filename and os.path.isfile(filename)):
            return True
        return True

    def process(self):
        """!
        @brief Process parameters for operation
        """
        if(self.args.bulk):
            logging.pprint(f"Bulk mode operation chosen.", 4)
            if (self.args.mode.lower() == "train"):
                if (os.path.isdir(self.args.output) and not os.path.isfile(self.args.output)) and (os.path.isdir(self.args.dataset) and not os.path.isfile(self.args.dataset)):
                    corpus_list = os.listdir(self.args.dataset)
                    for corpus in corpus_list:
                        self.import_model(self.args.input)
                        logging.pprint(f"Training {self.args.input} with {corpus}", 2)
                        output_file_name = corpus
                        model_extension = ""
                        if "." in self.args.input:
                            model_extension = self.args.input.split(".")[-1]
                        self.train(f"{self.args.dataset}/{corpus}", self.args.seperator, f"{self.args.output}/{corpus}.{model_extension}", output_forced=True, bulk=True)
                else:
                    logging.pprint("In bulk training, output and dataset should be a directory.")
                    exit(1)

            elif (self.args.mode.lower() == "generate"):
                if (os.path.isdir(self.args.wordlist) and not os.path.isfile(self.args.wordlist)) and (os.path.isdir(self.args.input) and not os.path.isfile(self.args.input)):
                    model_list = os.listdir(self.args.input)
                    print(model_list)
                    for input in model_list:
                        logging.pprint(f"Generating from {self.args.input}/{input} to {self.args.wordlist}/{input}.txt", 2)
                        self.import_model(f"{self.args.input}/{input}")
                        model_base = input
                        if "." in self.args.input:
                            model_base = input.split(".")[1]
                        self.generate(f"{self.args.wordlist}/{model_base}.txt", bulk=True)
                else:
                    logging.pprint("In bulk generation, input and wordlist should be directory.")

        else:
            self.import_model(self.args.input)
            if (self.args.mode.lower() == "generate"):
                self.generate(self.args.wordlist)


            elif (self.args.mode.lower() == "train"):
                self.train(self.args.dataset, self.args.seperator, self.args.output, output_forced=True)


            elif(self.args.mode.lower() == "combine"):
                self.train(self.args.dataset, self.args.seperator, self.args.output)
                self.generate(self.args.wordlist)


            else:
                logging.pprint("Invalid mode arguement given.")
                logging.pprint("Accepted modes: 'Generate', 'Train', 'Combine'")
                exit(5)

class AbstractGenerationModelCLI(BaseCLI):
    """!
    @brief abstract class for generation capable models
    @belongsto Python::Markopy
    @extends Python::Markopy::BaseCLI
    """
    @abstractmethod
    def add_arguments(self):
        "Add command line arguements to the parser"
        super().add_arguments()
        self.parser.add_argument("input",                            help="Input model file. This model will be imported before starting operation.")
        self.parser.add_argument("-w", "--wordlist",                 help="Wordlist file path to export generation results to. Will be ignored for training mode")
        self.parser.add_argument("--min", default=6,                 help="Minimum length that is allowed during generation")
        self.parser.add_argument("--max", default=12,                help="Maximum length that is allowed during generation")
        self.parser.add_argument("-n", "--count",                    help="Number of lines to generate. Ignored in training mode.")


class AbstractTrainingModelCLI(AbstractGenerationModelCLI, BaseCLI):
    """!
    @brief abstract class for training capable models
    @belongsto Python::Markopy
    @extends Python::Markopy::BaseCLI
    @extends Python::Markopy::AbstractGenerationModelCLI
    """
    @abstractmethod
    def add_arguments(self):
        "Add command line arguements to the parser"
        self.parser.add_argument("-o", "--output",                   help="Output model file. This model will be exported when done. Will be ignored for generation mode.")
        self.parser.add_argument("-d", "--dataset",                  help="Dataset file to read input from for training. Will be ignored for generation mode.")
        self.parser.add_argument("-s", "--seperator",                help="Seperator character to use with training data.(character between occurrence and value)")
        super().add_arguments()
