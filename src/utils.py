import time
import pickle
 
from datetime import datetime

def get_timestamp():
    return datetime.now()


def get_time():
    # datetime object containing current date and time
    now = datetime.now()

    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    
    return dt_string

def save(obj,name,mode="outputs"):
    """ Save an object as pickle
    Parameter:
        obj  : Object to save
        name : Filename
        mode : Where to save it inputs/output  
    """
    pickle.dump(obj,open( "../"+mode+"/"+name+".p", "wb" ))

def load(name,mode="inputs"):
    """ Loads an object from pickle
    Parameters:
        name: Filename
        mode: Where to load it
    """
    return pickle.load( open( "../"+mode+"/"+name+".p", "rb" ) )
    
