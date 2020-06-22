# get subject info

# values: 
# subjectnumber
# sex
# age
# group


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

data_dir='/jukebox/norman/amennen/RT_prettymouth/data' #this is my study directory
bids_dir = data_dir + '/bids/Norman/Mennen/5516_greenEyes'
projectName='greenEyes'
raw_dir= data_dir + '/conquest'
file_name='participants.tsv'
columns=['participant_id', 'age', 'sex', 'group']
data=[]
# script should just loop over all subjects possible
allsubjects = np.array([101, 102])
allsubject_groups = ['P', 'C'] # which context was heard first -- paranoi (P) or cheating (C)
allsubject_Names = ['0218191', '0219191']
nsub = len(allsubjects)

for s in np.arange(nsub):
	subjectNum=allsubjects[s]
	bids_id = 'sub-{0:03d}'.format(subjectNum)
	dicom_out= raw_dir + '/' + allsubject_Names[s] + '_' + projectName + '*/dcm'
	
	group= allsubject_groups[s]
	

	# go to an example file name for that person
	dicomfile= dicom_out + '/1-1-1.dcm'
	fn = glob.glob(dicomfile)[0]
	d = pydicom.read_file(fn)
	subjectAge = int(d.PatientAge[0:-1])
	subjectSex = d.PatientSex
	data.append((bids_id,subjectAge,subjectSex,group))

df = pd.DataFrame(data=data,columns=columns)
df.to_csv(join(bids_dir, 'participants.tsv'), sep='\t', index=False)

#df = df.append({'paticipant_id':bids_id,'age':subjectAge,'sex':subjectSex,'group':group})
