import os
import glob
from shutil import copyfile
import pandas as pd
import json
import numpy as np
from subprocess import call
# Purpose: make fake real-time runs by splitting all TRs
# Want to go through each run of the story task and make separate folder

bids_dir='/jukebox/norman/amennen/RT_prettymouth/data/bids/Norman/Mennen/5516_greenEyes'
code_dir=bids_dir + '/code'
split_dir=bids_dir + '/derivatives/RTCHECK'

subjects = np.array([101,102])
ses=1
for subjectNum in subjects:
	bids_id = 'sub-{0:03d}'.format(subjectNum)
	ses_id = 'ses-{0:02d}'.format(ses)

	func_dir='{0}/{1}/{2}/func/*task-story*.nii.gz'.format(bids_dir,bids_id,ses_id)
	all_func_files = glob.glob(func_dir)
	nruns = len(all_func_files)

	for f in np.arange(nruns):
		this_file = all_func_files[f].split('/')[-1]
		this_file_base = this_file[0:-7]

		# make new directory to split files into
		new_func_dir='{0}/{1}/{2}/{3}'.format(split_dir,bids_id,ses_id,this_file_base)
		command = 'mkdir -pv {0}'.format(new_func_dir)
		call(command,shell=True)
		os.chdir(new_func_dir)


		command = 'fslsplit {0}'.format(all_func_files[f]) 
		call(command,shell=True)
		
		os.chdir(code_dir)