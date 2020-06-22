# PURPOSE: take new nifti file and register it to MNI BOLD to be masked and preprocessed
# Steps
# 1. MCflirt --> output transformation matrix for that volume

import os
import glob
from shutil import copyfile
import pandas as pd
import json
import numpy as np
from subprocess import call
import time

# TO DO: check again that doing for a couple TRs is the right reference file between everything including the affine (or it just could be poor mcflirt alignment)

bids_dir='/jukebox/norman/amennen/RT_prettymouth/data/bids/Norman/Mennen/5516_greenEyes'
code_dir=bids_dir + '/code'
split_dir=bids_dir + '/derivatives/RTCHECK'
MNI_ref_BOLD='/jukebox/norman/amennen/MNI_things/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_BOLD_brain.nii.gz'

# convert dicom to nifti
subjects = np.array([101,102])
ses=1
nsubs = len(subjects)
all_tr_time = np.zeros((4,nsubs))
this_tr_reg = np.zeros(())
for s in np.arange(nsubs): 
	subjectNum = subjects[s]
	bids_id = 'sub-{0:03d}'.format(subjectNum)
	wf_dir=bids_dir+'/derivatives/work/fmriprep_wf/single_subject_{0:03d}_wf'.format(subjectNum)
	bids_dir+'/derivatives/fmriprep/sub_{0:03d}/ses_{0:02d}/func/'.format(subjectNum)
	# this is a skull stripped image-- shouldn't be using it unless we're also going to do brain extraction on new images coming in
	#ref_BOLD=glob.glob(wf_dir+ '/func_preproc_ses_01_task_story_run_01_wf/bold_reference_wf/enhance_and_skullstrip_bold_wf/apply_mask/uni_xform_masked.nii.gz')[0]
	ref_BOLD=glob.glob(wf_dir + '/func_preproc_ses_01_task_story_run_01_wf/bold_reference_wf/gen_ref/ref_image.nii.gz')[0]
	BOLD_to_T1=wf_dir + '/func_preproc_ses_01_task_story_run_01_wf/bold_reg_wf/bbreg_wf/fsl2itk_fwd/affine.txt'
	T1_to_MNI= wf_dir + '/anat_preproc_wf/t1_2_mni/ants_t1_to_mniComposite.h5'

	ses_id = 'ses-{0:02d}'.format(ses)

	func_dir='{0}/{1}/{2}/func/*task-story*.nii.gz'.format(bids_dir,bids_id,ses_id)
	all_func_files = glob.glob(func_dir)
	nruns = len(all_func_files)

	for f in np.arange(nruns):
		this_file = all_func_files[f].split('/')[-1]
		this_file_base = this_file[0:-7]
		split_func_dir='{0}/{1}/{2}/{3}'.format(split_dir,bids_id,ses_id,this_file_base)

		new_tr_files = glob.glob(split_func_dir + '/vol*nii.gz')
		nTRs = len(new_tr_files)
		this_tr_reg = np.zeros((nTRs,))
		for t in np.arange(nTRs):
			this_tr_file = new_tr_files[t].split('/')[-1]
			this_tr_base = this_tr_file[0:-7]
			# make new folder for this specific TR
			new_tr_dir = '{0}/{1}'.format(split_func_dir,this_tr_base)
			command = 'mkdir -pv {0}'.format(new_tr_dir)
			call(command,shell=True)
			#os.chdir(new_tr_dir)

			# now for this file run flirt (flirt bc could have weird contrast issues involved with assuming it's from the same run)

			# command = 'flirt -in {0} -ref {1} -o {2}/new2ref_flirt.nii.gz -omat {2}/new2ref_flirt.mat -dof 6 -noresample'.format(new_tr_files[t],ref_BOLD,new_tr_dir)
			# t1 = time.time()
			# call(command,shell=True)
			# t2 = time.time()
			# print('elapsed time = {0}'.format(t2-t1))

			command = 'mcflirt -in {0} -reffile {1} -out {2}/new2ref_MC -mats'.format(new_tr_files[t],ref_BOLD,new_tr_dir)
			#command = 'mcflirt -in {0} -reffile {1} -out {2}/new2ref_MC_NN -nn_final -mats'.format(new_tr_files[t],ref_BOLD,new_tr_dir)
			#command = 'mcflirt -in {0} -reffile {1} -out {2}/new2ref_MC_sinc -sinc_final -mats'.format(new_tr_files[t],ref_BOLD,new_tr_dir)
			#command = 'mcflirt -in {0} -reffile {1} -out {2}/new2ref_MC_spline -spline_final -mats'.format(new_tr_files[t],ref_BOLD,new_tr_dir)

			tr_start = time.time()
			t1 = time.time()
			call(command,shell=True)
			t2 = time.time()
			print('elapsed time = {0}'.format(t2-t1))

			# now run the command to convert this
			command = 'c3d_affine_tool -ref {0} -src {1} {2}/new2ref_MC.mat/MAT_0000 -fsl2ras -oitk {2}/new2ref.txt'.format(ref_BOLD,new_tr_files[t],new_tr_dir)
			#command = 'c3d_affine_tool -ref {0} -src {1} {2}/new2ref_MC_spline.mat/MAT_0000 -fsl2ras -oitk {2}/new2ref_spline.txt'.format(ref_BOLD,new_tr_files[t],new_tr_dir)

			t1 = time.time()
			call(command,shell=True)
			t2 = time.time()
			print('elapsed time = {0}'.format(t2-t1))

			# now combine all transforms to move to MNI space
			command = 'antsApplyTransforms --default-value 0 --float 1 --interpolation LanczosWindowedSinc -d 3 -e 3 --input {0} --reference-image {1} --output {2}/{3}_spline_space-MNI.nii.gz --transform {4}/new2ref_spline.txt --transform {5} --transform {6} -v 1'.format(new_tr_files[t],MNI_ref_BOLD,new_tr_dir,this_tr_base,new_tr_dir,BOLD_to_T1,T1_to_MNI)
			t1 = time.time()
			call(command,shell=True)
			t2 = time.time()
			tr_stop = time.time()
			this_tr_reg[t] = tr_stop - tr_start
			print('elapsed time = {0}'.format(t2-t1))
		all_tr_time[f,s] = np.mean(this_tr_reg)
np.save('registration_timing_spline', all_tr_time)


# new: try with ants registration -- rigid -- link here: https://github.com/ANTsX/ANTs/wiki/Anatomy-of-an-antsRegistration-call - rigid optimal is between 0.1-0.25 so increase to help with speed
# command = 'antsRegistration --dimensionality 3 --float 1 --output {0}/antsReg_new2ref  --interpolation Linear --winsorize-image-intensities [0.005,0.995] --use-histogram-matching 1 --transform Rigid[0.1] --metric MI[{1},{2},1,32]'.format(new_tr_dir,ref_BOLD,new_tr_files[t])
# command = 'antsRegistration -d 3 -r [{0},{1}, 1] -m Mattes[{0}, {1}, 1, 32, regular, 0.2] -t Rigid[0.25] -c 100x20 -s 1x0 -f 2x1 -u 1 -z 1 -l 1 -o [{4}/antsReg_new2ref, {5}/antsReg_new2ref.nii.gz] -x [NA,NA]'.format(ref_BOLD,new_tr_files[t], ref_BOLD, new_tr_files[t], new_tr_dir,new_tr_dir)
# command = 'antsMotionCorr -d 3 -l 1 -m MI[{0}, {1}, 1, 32, Regular, 0.2] -u 1 -t Rigid[0.25] -i 15x3 -f 2x1 -s 1x0 -w 1 -e 1 {2}/antsReg_new2ref, {3}/antsReg_new2ref.nii.gz]'.format(ref_BOLD,new_tr_files[t],new_tr_dir,new_tr_dir)
# #command = 'antsRegistration -d 3 -f 2x1 --float 1 -r [ {0}, {1}, 1] -t Rigid[0.25] -m MI[ {2}, {3}, 1, 32] -s 1x0 -c 100x20 -u 0 -z 1 -l 1 -v 1 -o [{4}/antsReg_new2ref, {5}/antsReg_new2ref.nii.gz]'.format(ref_BOLD,new_tr_files[t], ref_BOLD, new_tr_files[t], new_tr_dir,new_tr_dir)
# t1 = time.time()
# call(command,shell=True)
# t2 = time.time()
# print('elapsed time = {0}'.format(t2-t1))

# command = 'ANTS 3 -o {0}/antsReg_new2ref -m MI[{1},{2},1,32] -i 0 --rigid-affine true  --use-histogram-matching true'.format(new_tr_dir,ref_BOLD,new_tr_files[t])
# t1 = time.time()
# call(command,shell=True)
# t2 = time.time()
# print('elapsed time = {0}'.format(t2-t1))