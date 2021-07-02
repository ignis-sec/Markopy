


##
# @file evaluate.py
# @brief Evaluation of model integrity and score
#
from abc import abstractmethod
import re
import allogate as logging
import statistics
import os
from copy import copy
import glob
import re

class Evaluator:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.checks = []
        self.TEST_PASS_SYMBOL = "✅"
        self.TEST_FAIL_SYMBOL = "❌"
        self.all_checks_passed = True
        self.files = []
        if("*" in filename):
            self.files = glob.glob(filename)
        else:
            self.files.append(filename)
        return True

    def evaluate(self) -> bool:
        for file in self.files:
            self._evaluate(file)
        
        self.check_funcs = [func for func in dir(self) if (callable(getattr(self, func)) and func.startswith("check_"))]

    @abstractmethod
    def _evaluate(self, file) -> list:
        if(not os.path.isfile(file)):
            logging.pprint(f"Given file {file} is not a valid filename")
            return False
        else:
             return open(file, "rb").read().split(b"\n")
        
        

    def success(self, checkname):
        self.checks.append((checkname, self.TEST_PASS_SYMBOL))

    def fail(self, checkname):
        self.all_checks_passed = False
        self.checks.append((checkname, self.TEST_FAIL_SYMBOL))
        self.checks = []

    def finalize(self):
        print("\n################ Checks ################ ")
        for test in self.checks:
            logging.pprint(f"{test[0]:30}:{test[1]} ")
        print("\n")

        return self.all_checks_passed

class ModelEvaluator(Evaluator):
    def __init__(self, filename: str) -> None:
        
        valid = super().__init__(filename)
        
        if not valid:
            return False

    def evaluate(self):
        logging.VERBOSITY=2
        logging.SHOW_STACK_THRESHOLD=3
        super().evaluate()
        for file in self.files:
            logging.pprint(f"Model: {file.split('/')[-1]}: ",2)
            edges = super()._evaluate(file)
            if not edges:
                continue
            self.lnodes = {}
            self.rnodes = {}
            self.ews = []
            self.edge_count = len(edges)
            for edge in edges:
                if(edge ==b''):
                    self.edge_count-=1
                    continue
                try:
                    e = edge.split(b',')
                    self.ews.append(int(edge[2:-2:1]))
                    if(e[0] not in self.lnodes):
                        self.lnodes[e[0]]=1
                    else:
                        self.lnodes[e[0]]+=1
                    if(e[-1] not in self.rnodes):
                        self.rnodes[e[-1]]=1
                    else:
                        self.rnodes[e[-1]]+=1
                except Exception as e:
                    print(e)
                    logging.pprint(f"Model file is corrupted.", 0)
                    continue
        
            self.lnode_count = len(self.lnodes)
            self.rnode_count = len(self.rnodes)
            logging.pprint(f"total edges: {self.edge_count}", 1)
            logging.pprint(f"unique left nodes: {self.lnode_count}", 1)
            logging.pprint(f"unique right nodes: {self.rnode_count}", 1)

            for check in self.check_funcs:
                try:
                    self.__getattribute__(check)()
                except Exception as e:
                    print(e)
                    self.fail(f"Exceptionn in {check}")
            self.finalize()
        
    def check_dangling(self):
        if(self.lnode_count == self.rnode_count):
            self.success("No dangling nodes")
        else:
            logging.pprint(f"Dangling nodes found, lnodes and rnodes do not match", 0)
            self.fail("No dangling nodes")
    
    def check_structure(self):
        if((self.lnode_count-1) * (self.rnode_count-1) + 2*(self.lnode_count-1)):  
            self.success("Model structure")
        else:
            logging.pprint(f"Model did not satisfy structural integrity check (lnode_count-1) * (rnode_count-1) + 2*(lnode_count-1)", 0)
            self.fail("Model structure")

    def check_weight_deviation(self):
        mean = sum(self.ews) / len(self.ews)
        variance = sum([((x - mean) ** 2) for x in self.ews]) / len(self.ews)
        res = variance ** 0.5
        self.stdev = res
        if(res==0):
            logging.pprint(f"Model seems to be untrained", 0)
            self.fail("Model has any training")
        else:
            self.success("Model has any training")
        if(res<3000):
            logging.pprint(f"Model is not adequately trained. Might result in inadequate results", 1)
            self.fail("Model has training")
            self.fail(f"Model training score: {round(self.stdev,2)}")
        else:
            self.success("Model has training")
            self.success(f"Model training score: {round(self.stdev)}")

    def check_min(self):
        count = 0
        for ew in self.ews:
            if ew==0:
                count+=1
        if(count > self.rnode_count*0.8):
            self.fail("Too many 0 edges")
            logging.pprint(f"0 weighted edges are dangerous and may halt the model.", 0)
        else:
            self.success("0 edges below threshold")
    
    def check_min_10percent(self):
        sample = self.ews[int(self.edge_count*0.1)]
        #print(f"10per: {sample}")
        avg = sum(self.ews) / len(self.ews)
        #print(f"avg: {avg}")
        med = statistics.median(self.ews)
        #print(f"med: {med}")

    def check_lean(self):
        sample = self.ews[int(self.edge_count*0.1)]
        avg = sum(self.ews) / len(self.ews)
        med = statistics.median(self.ews)
        
        if(med*10<sample):
            logging.pprint("Median is too left leaning and might indicate high entropy")
            self.fail("Median too left leaning")
        else:
            self.success("Median in expected ratio")
        pass

        if(sample*5>avg):
            logging.pprint("Least probable 10% too close to average, might indicate inadequate training")
            self.fail("Bad bottom 10%")
        else:
            self.success("Good bottom 10%")
        pass
        

    def check_distrib(self):
        sorted_ews = copy(self.ews)
        sorted_ews.sort(reverse=True)
        ratio1 = sorted_ews[0]/sorted_ews[int(self.edge_count/2)]
        ratio2 = sorted_ews[int(self.edge_count/2)]/sorted_ews[int(self.edge_count*0.1)]
        #print(ratio1)
        #print(ratio2)


class CorpusEvaluator(Evaluator):
    def __init__(self, filename: str) -> None:
        valid = super().__init__(filename)
        if not valid:
            return False

    def evaluate(self):
        logging.pprint("WARNING: This takes a while with larger corpus files", 2)
        logging.VERBOSITY=2
        logging.SHOW_STACK_THRESHOLD=3
        super().evaluate()
        for file in self.files:

            delimiter = ''
            sum=0
            max=0
            total_chars = 0
            lines_count = 0
            bDelimiterConflict=False
            logging.pprint(f"Corpus: {file.split('/')[-1]}: ",2)
            with open(file, "rb") as corpus:
                for line in corpus:
                    lines_count+=1
                    match = re.match(r"([0-9]+)(.)(.*)\n", line.decode()).groups()
                    if(delimiter and delimiter!=match[1]):
                        bDelimiterConflict = True
                        
                    elif(not delimiter):
                        delimiter = match[1]
                        logging.pprint(f"Delimiter is: {delimiter.encode()}")
                    sum +=int(match[0])
                    total_chars += len(match[2])
                    if(int(match[0])>max):
                        max=int(match[0])

                if(bDelimiterConflict):
                    self.fail("Incorrect delimiter found")
                else:
                    self.success("No structural conflicts")

                logging.pprint(f"Total number of lines: {lines_count}")
                logging.pprint(f"Sum of all string weights: {sum}")
                logging.pprint(f"Character total: {total_chars}")
                logging.pprint(f"Average length: {total_chars/lines_count}")
                logging.pprint(f"Average weight: {sum/lines_count}")

            self.finalize()
    def _evaluate(self, file) -> list:
        if(not os.path.isfile(file)):
            logging.pprint(f"Given file {file} is not a valid filename")
            return False
        else:
            return True
            