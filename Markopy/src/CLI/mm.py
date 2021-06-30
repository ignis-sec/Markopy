

##
# @file mm.py
# @brief Abstract representation of CPP/Python intermediate layer classes.
#

from abc import abstractmethod

from importer import import_markopy
markopy = import_markopy()

class MarkovModel(markopy.MarkovPasswords):
    """!
    @brief Abstract representation of a markov model
    @implements Markov::API::MarkovPasswords
    @belongsto Python::Markopy

    To help with the python-cpp gateway documentation.
    """
    @abstractmethod
    def Import(filename : str):
        pass

    @abstractmethod
    def Export(filename : str):
        pass

    @abstractmethod
    def Train(dataset: str, seperator : str, threads : int):
        pass

    @abstractmethod
    def Generate(count : int, wordlist : str, minlen : int, maxlen: int, threads : int):
        pass


class ModelMatrix(markopy.ModelMatrix):
    """!
    @brief Abstract representation of a matrix based model
    @implements Markov::API::ModelMatrix 
    @belongsto Python::Markopy

    To help with the python-cpp gateway documentation.
    """

    @abstractmethod
    def FastRandomWalk(count : int, wordlist : str, minlen : int, maxlen : int):
        pass
