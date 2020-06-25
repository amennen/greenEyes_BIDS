# get subject info

# values: 
# subjectnumber
# sex
# age
# group
# experiment

import os
from os.path import exists, join
from os import makedirs
from glob import glob
from shutil import copyfile
import pandas as pd
import nibabel as nib
import json
import pydicom
import pandas as pd
import numpy as np
import glob

def getSubjectInterpretation(subject_num):
    # load interpretation file and get it
    # will be saved in subject full day path
    bids_id = 'sub-{0:03d}'.format(subject_num)
    ses_id = 'ses-{0:02d}'.format(2)
    filename = '/jukebox/norman/amennen/RT_prettymouth/data/intelData/' + bids_id + '/' + ses_id + '/' + bids_id + '_' + ses_id + '_' + 'intepretation.txt'
    z = open(filename, "r")
    temp_interpretation = z.read()
    if 'C' in temp_interpretation:
        interpretation = 'C'
    elif 'P' in temp_interpretation:
        interpretation = 'P'
    return interpretation

data_dir='/jukebox/norman/amennen/RT_prettymouth/data' #this is my study directory
bids_dir = data_dir + '/bids/Norman/Mennen/5516_greenEyes'
projectName='greenEyes'
raw_dir= data_dir + '/conquest'
file_name='participants2.tsv'
columns=['participant_id', 'sex', 'age', 'group', 'experiment']
data=[]
# script should just loop over all subjects possible
subjects_exp1 = [2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19]

subjects_exp2 = [25,26,28,29,30,31,32,33,35,36,37,38,39,41,40,42,43,44,45,46]
all_subjects = subjects_exp1 + subjects_exp2
#allsubject_groups = ['P', 'C'] # which context was heard first -- paranoi (P) or cheating (C)
#allsubject_Names = ['0218191', '0219191']
nsub = len(all_subjects)
original_tsv = bids_dir + '/' + 'participants.tsv'
original_data = pd.read_csv(original_tsv, sep='\t')
participant_id = original_data['participant_id']
age = original_data['age']
sex = original_data['sex']
for s in np.arange(nsub):
	subjectNum=all_subjects[s]
	bids_id = 'sub-{0:03d}'.format(subjectNum)
	subjRow = np.argwhere(original_data['participant_id'] == bids_id)[0][0]  
	subjAge = age[subjRow]
	subjSex = sex[subjRow]
	subjExp = 'nan'
	if subjectNum in subjects_exp1:
		subjExp = '1'
	elif subjectNum in subjects_exp2:
		subjExp = '2'
	subjGroup = getSubjectInterpretation(subjectNum)
	data.append((bids_id,subjSex,subjAge,subjGroup, subjExp))

df = pd.DataFrame(data=data,columns=columns)
df.to_csv(join(bids_dir, file_name), sep='\t', index=False)

#df = df.append({'paticipant_id':bids_id,'age':subjectAge,'sex':subjectSex,'group':group})
