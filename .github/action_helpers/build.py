#!/ur/bin/python3

# This is a build script for the github workflows. You can use it for building the project too.
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("step", help="Step to build")
parser.add_argument("-p", "--pyversion", help="Python version to use")

args = parser.parse_args() 

def decide_action_type(ref,trigger):
    print(f"ref: {ref}, trigger: {trigger}")
    if(ref == "refs/heads/main" and trigger == "push"):
        return "minor"

    elif(ref == "refs/heads/dev" and trigger == "push"):
        return "patch"
    elif(ref == "refs/heads/dev" and trigger == "pull_request"):
        return "dev"


def set_action_outputs(**kwargs):
    for key in kwargs:
        val = kwargs[key]
        print(f"Setting output for:{key}.")
        print(f"Value: {val}")
        print(f"::set-output name={key}::{val}")


if __name__ == "__main__":
    print(os.environ)
    ref     = os.environ["GITHUB_REF"]
    trigger = os.environ["GITHUB_EVENT_NAME"] 
    a_type = decide_action_type(ref, trigger)
    print(f"Workflow configuration set to {a_type}.")
    
    
    if(args.step == "version_bump"):
        if(a_type == "minor"):
            set_action_outputs(bump="minor", branch="main", prerelease="false", draft="false")

        elif(a_type == "patch"):
            set_action_outputs(bump="patch", branch="<junk>", prerelease="true", draft="false")

        elif(a_type == "dev"):
            set_action_outputs(bump="patch", branch="<junk>", prerelease="false", draft="true")
        else:
            set_action_outputs(bump="patch", branch="<junk>", prerelease="false", draft="true")

    if(args.step == "version_matrix"):
        
        if(a_type == "minor"):
            lm = "[3.6, 3.7, 3.8, 3.9]"
            wm = "[3.8, 3.9]"

        elif(a_type == "patch"):
            lm = "[3.8]"
            wm = "[3.8]"

        else:
            lm = "[3.8]"
            wm = "[3.8]"

        set_action_outputs(linux_matrix=lm, windows_matrix=wm)  

    if(args.step == "linux_python_libs"):
        os.system(f".github/action_helpers/actions_linux_python_libs.sh {args.pyversion}")

    if(args.step == "linux_setup_boost"):
        os.system(f".github/action_helpers/actions_linux_setup_boost.sh {args.pyversion}")
    
    if(args.step == "linux_setup_libboost"):
        os.system(f".github/action_helpers/actions_linux_setup_libboost.sh {args.pyversion}")


