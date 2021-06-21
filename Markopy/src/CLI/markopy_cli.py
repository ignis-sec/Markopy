import markopy
import argparse
import allogate as logging
import re
import os

parser = argparse.ArgumentParser(description="Python wrapper for MarkovPasswords.",
epilog=f"""Sample runs:
{__file__} train untrained.mdl -d dataset.dat -s "\\t" -o trained.mdl
    Import untrained.mdl, train it with dataset.dat which has tab delimited data, output resulting model to trained.mdl\n

{__file__} generate trained.mdl -n 500 -w output.txt
    Import trained.mdl, and generate 500 lines to output.txt

{__file__} combine untrained.mdl -d dataset.dat -s "\\t" -n 500 -w output.txt
    Train and immediately generate 500 lines to output.txt. Do not export trained model.

{__file__} combine untrained.mdl -d dataset.dat -s "\\t" -n 500 -w output.txt -o trained.mdl
    Train and immediately generate 500 lines to output.txt. Export trained model.
""", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("mode",                             help="Operation mode, supported modes: \"generate\", \"train\" and \"combine\".")
parser.add_argument("input",                            help="Input model file. This model will be imported before starting operation.")
parser.add_argument("-o", "--output",                   help="Output model file. This model will be exported when done. Will be ignored for generation mode.")
parser.add_argument("-d", "--dataset",                  help="Dataset file to read input from for training. Will be ignored for generation mode.")
parser.add_argument("-s", "--seperator",                help="Seperator character to use with training data.(character between occurrence and value)")
parser.add_argument("-w", "--wordlist",                 help="Wordlist file path to export generation results to. Will be ignored for training mode")
parser.add_argument("--min", default=6,                 help="Minimum length that is allowed during generation")
parser.add_argument("--max", default=12,                help="Maximum length that is allowed during generation")
parser.add_argument("-n", "--count",                    help="Number of lines to generate. Ignored in training mode.")
parser.add_argument("-t", "--threads",default=10,       help="Number of lines to generate. Ignored in training mode.")
parser.add_argument("-v", "--verbosity",action="count", help="Output verbosity.")
parser.add_argument("-b", "--bulk",action="store_true", help="Bulk generate or bulk train every corpus/model in the folder.")
args = parser.parse_args() 




def cli_init(input_model):
    logging.VERBOSITY = 0
    if args.verbosity:
        logging.VERBOSITY = args.verbosity
        logging.pprint(f"Verbosity set to {args.verbosity}.", 2)

    logging.pprint("Initializing model.", 1)
    model = markopy.ModelMatrix()
    logging.pprint("Model initialized.", 2)

    logging.pprint("Importing model file.", 1)

    if(not os.path.isfile(input_model)):
        logging.pprint(f"Model file at {input_model} not found. Check the file path, or working directory")
        exit(1)

    model.Import(input_model)
    logging.pprint("Model imported successfully.", 2)
    return model

def cli_train(model, dataset, seperator, output, output_forced=False, bulk=False):
    if not (dataset and seperator and (output or not output_forced)):
        logging.pprint(f"Training mode requires -d/--dataset{', -o/--output' if output_forced else''} and -s/--seperator parameters. Exiting.")
        exit(2)

    if(not bulk and not os.path.isfile(dataset)):
        logging.pprint(f"{dataset} doesn't exists. Check the file path, or working directory")
        exit(3)

    if(output and os.path.isfile(output)):
        logging.pprint(f"{output} exists and will be overwritten.",1 )

    if(seperator == '\\t'):
        logging.pprint("Escaping seperator.", 3)
        seperator = '\t'
    
    if(len(seperator)!=1):
        logging.pprint(f'Delimiter must be a single character, and "{seperator}" is not accepted.')
        exit(4)

    logging.pprint(f'Starting training.', 3)
    model.Train(dataset,seperator, int(args.threads))
    logging.pprint(f'Training completed.', 2)

    if(output):
        logging.pprint(f'Exporting model to {output}', 2)
        model.Export(output)
    else:
        logging.pprint(f'Model will not be exported.', 1)

def cli_generate(model, wordlist, bulk=False):
    if not (wordlist or args.count):
        logging.pprint("Generation mode requires -w/--wordlist and -n/--count parameters. Exiting.")
        exit(2)
    
    if(bulk and os.path.isfile(wordlist)):
        logging.pprint(f"{wordlist} exists and will be overwritten.", 1)
    model.Generate(int(args.count), wordlist, int(args.min), int(args.max), int(args.threads))


if(args.bulk):
    logging.pprint(f"Bulk mode operation chosen.", 4)

    if (args.mode.lower() == "train"):
        if (os.path.isdir(args.output) and not os.path.isfile(args.output)) and (os.path.isdir(args.dataset) and not os.path.isfile(args.dataset)):
            corpus_list = os.listdir(args.dataset)
            for corpus in corpus_list:
                model = cli_init(args.input)
                logging.pprint(f"Training {args.input} with {corpus}", 2)
                output_file_name = corpus
                model_extension = ""
                if "." in args.input:
                    model_extension = args.input.split(".")[-1]
                cli_train(model, f"{args.dataset}/{corpus}", args.seperator, f"{args.output}/{corpus}.{model_extension}", output_forced=True, bulk=True)
        else:
            logging.pprint("In bulk training, output and dataset should be a directory.")
            exit(1)

    elif (args.mode.lower() == "generate"):
        if (os.path.isdir(args.wordlist) and not os.path.isfile(args.wordlist)) and (os.path.isdir(args.input) and not os.path.isfile(args.input)):
            model_list = os.listdir(args.input)
            print(model_list)
            for input in model_list:
                logging.pprint(f"Generating from {args.input}/{input} to {args.wordlist}/{input}.txt", 2)
                
                model = cli_init(f"{args.input}/{input}")
                model_base = input
                if "." in args.input:
                    model_base = input.split(".")[1]
                cli_generate(model, f"{args.wordlist}/{model_base}.txt", bulk=True)
        else:
            logging.pprint("In bulk generation, input and wordlist should be directory.")

else:
    model = cli_init(args.input)
    if (args.mode.lower() == "generate"):
        cli_generate(model, args.wordlist)


    elif (args.mode.lower() == "train"):
        cli_train(model, args.dataset, args.seperator, args.output, output_forced=True)


    elif(args.mode.lower() == "combine"):
        cli_train(model, args.dataset, args.seperator, args.output)
        cli_generate(model, args.wordlist)


    else:
        logging.pprint("Invalid mode arguement given.")
        logging.pprint("Accepted modes: 'Generate', 'Train', 'Combine'")
        exit(5)