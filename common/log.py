import logging
import os
from datetime import datetime

def init_log(path:str):
    	
    # create unique directories
	now = datetime.now().strftime("%b-%d_%H-%M-%S")  
	save_path = os.path.join(path, now)
	os.makedirs(save_path, exist_ok=True)
	log_file = os.path.join(save_path, 'out.log')
	# create logger
	logger = logging.getLogger('sur')
	logger.setLevel(logging.INFO)
	fh = logging.FileHandler(log_file, mode='w')
	fh.setLevel(logging.INFO)
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s, %(levelname)s: %(message)s')
	fh.setFormatter(formatter)
	ch.setFormatter(formatter)
	logger.addHandler(fh)
	logger.addHandler(ch)
	return logger, save_path, log_file