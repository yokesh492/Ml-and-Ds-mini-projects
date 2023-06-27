import os
import sys

import numpy as np 
import pandas as pd
import dill

from src.exception import CustomException

def save_object(filepath,object):
    try:
        dir_path = os.path.dirname(filepath)#artifacts
        os.makedirs(dir_path,exist_ok=True)
        with open(filepath, 'w') as f_obj:
            dill.dump(object,f_obj)
        
    except Exception as e:
        CustomException(e,sys)

