import time
 
from datetime import datetime

def get_timestamp():
    return datetime.now()


def get_time():
    # datetime object containing current date and time
    now = datetime.now()

    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    
    return dt_string