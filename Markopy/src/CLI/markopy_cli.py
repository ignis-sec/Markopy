import markopy
import argparse
import allogate as logging
import re
import os

parser = argparse.ArgumentParser(description="Python wrapper for MarkovPasswords.",
epilog=f"""Sample runs:
{__file__} train -i untrained.mdl -d dataset.dat -s "\\t" -o trained.mdl
    Import untrained.mdl, train it with dataset.dat which has tab delimited data, output resulting model to trained.mdl\n

{__file__} generate -i trained.mdl -n 500 -w output.txt
    Import trained.mdl, and generate 500 lines to output.txt

{__file__} combine -i untrained.mdl -d dataset.dat -s "\\t" -n 500 -w output.txt
    Train and immediately generate 500 lines to output.txt. Do not export trained model.

{__file__} combine -i untrained.mdl -d dataset.dat -s "\\t" -n 500 -w output.txt -o trained.mdl
    Train and immediately generate 500 lines to output.txt. Export trained model.
""", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("mode",             help="Operation mode, supported modes: \"generate\", \"train\" and \"combine\".")
parser.add_argument("input",            help="Input model file. This model will be imported before starting operation.")
parser.add_argument("-o", "--output",   help="Output model file. This model will be exported when done. Will be ignored for generation mode.")
parser.add_argument("-d", "--dataset",  help="Dataset file to read input from for training. Will be ignored for generation mode.")
parser.add_argument("-s", "--seperator",help="Seperator character to use with training data.(character between occurrence and NodeValue)")
parser.add_argument("-w", "--wordlist", help="Wordlist file path to export generation results to. Will be ignored for training mode")
parser.add_argument("-n", "--count",    help="Number of lines to generate. Ignored in training mode.")
parser.add_argument("-v", "--verbosity",action="count", help="Output verbosity.")
args = parser.parse_args() 




def cli_init():
    logging.VERBOSITY = 0
    if args.verbosity:
        logging.VERBOSITY = args.verbosity
        logging.pprint(f"Verbosity set to {args.verbosity}.", 2)

    logging.pprint("Initializing model.", 1)
    model = markopy.MarkovPasswords()
    logging.pprint("Model initialized.",2)

    logging.pprint("Importing model file.", 1)

    if(not os.path.isfile(args.input)):
        logging.pprint(f"Model file at {args.input} not found. Check the file path, or working directory")
        exit(1)

    model.Import(args.input)
    logging.pprint("Model imported successfully.", 2)
    return model

def cli_train(model, output_forced=False):
    if not (args.dataset and args.seperator and (args.output or not output_forced)):
        logging.pprint(f"Training mode requires -d/--dataset{', -o/--output' if output_forced else''} and -s/--seperator parameters. Exiting.")
        exit(2)

    if(not os.path.isfile(args.dataset)):
        logging.pprint(f"{args.dataset} doesn't exists. Check the file path, or working directory")
        exit(3)

    if(args.output and os.path.isfile(args.output)):
        logging.pprint(f"{args.output} exists and will be overwritten.",1 )

    if(args.seperator == '\\t'):
        logging.pprint("Escaping seperator.", 3)
        args.seperator = '\t'
    
    if(len(args.seperator)!=1):
        logging.pprint(f'Delimiter must be a single character, and "{args.seperator}" is not accepted.')
        exit(4)

    logging.pprint(f'Starting training.', 3)
    model.Train(args.dataset,args.seperator)
    logging.pprint(f'Training completed.', 2)

    if(args.output):
        logging.pprint(f'Exporting model.', 2)
        model.Export(args.output)
    else:
        logging.pprint(f'Model will not be exported.', 1)

def cli_generate(model):
    if not (args.wordlist and args.count):
        logging.pprint("Generation mode requires -w/--wordlist and -n/--count parameters. Exiting.")
        exit(2)
    
    if(os.path.isfile(args.wordlist)):
        logging.pprint(f"{args.wordlist} exists and will be overwritten.", 1)

    model.Generate(int(args.count), args.wordlist,6,12)




model = cli_init()
if (args.mode.lower() == "generate"):
    cli_generate(model)


elif (args.mode.lower() == "train"):
    cli_train(model, output_forced=True)


elif(args.mode.lower() == "combine"):
    cli_train(model)
    cli_generate(model)


else:
    logging.pprint("Invalid mode arguement given.")
    logging.pprint("Accepted modes: 'Generate', 'Train', 'Combine'")
    exit(5)