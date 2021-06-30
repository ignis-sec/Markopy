
##
# @file importer.py
# @brief dynamic import wrapper for markopy model
#

from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader, ExtensionFileLoader
import os


def import_markopy():
    """! @brief import and return markopy module
         @returns markopy module
    """
    ext = "so"
    if os.name == 'nt':
        ext="pyd"
    try:
        spec = spec_from_loader("markopy", ExtensionFileLoader("markopy", os.path.abspath(f"markopy.{ext}")))
        markopy = module_from_spec(spec)
        spec.loader.exec_module(markopy)
        return markopy
    except ImportError as e:
        print(f"({__file__}) Working in development mode. Trying to load markopy.{ext} from ../../../out/")
        if(os.path.exists(f"../../../out/lib/markopy.{ext}")):
            spec = spec_from_loader("markopy", ExtensionFileLoader("markopy", os.path.abspath(f"../../../out/lib/markopy.{ext}")))
            markopy = module_from_spec(spec)
            spec.loader.exec_module(markopy)
            return markopy
        else:
            raise e