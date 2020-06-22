#! /bin/bash

set -e #stop immediately if error occurs

subj=$1
session=$2

data_dir=/jukebox/norman/amennen/RT_prettymouth/data #this is my study directory
raw_dir=$data_dir/conquest
extra_dir=$data_dir/extra
bids_dir=$data_dir/bids/Norman/Mennen/5516_greenEyes #this is where BIDS formatted data will end up and should match the program card on the scanner
scripts_dir=$bids_dir/code

# STEP 5 deface T1 (run on local)
# cd /Volumes/norman/emcdevitt/studies/SVD/data/bids/Norman/McDevitt/7137_viodiff
# code/deface.sh [sid] [session]

# STEP 6 -- cleanup

# delete scout images
find $bids_dir/sub-$subj/ses-$session -name "*scout*" -delete
find $bids_dir/sub-$subj/ses-$session -name "*dup*" -delete
# NEW -- ACM added 5/30/19 - if anat folder is now empty, delete anat folder
if [ $session -gt 1 ]
then
	# delete if folder empty after deleting files
	anat_dir=$bids_dir/sub-$subj/ses-$session/anat
	if [ "$(ls -A $anat_dir)" ]
	then
		echo "other anat files--not deleting folder"
	else
		rm -r $anat_dir
		echo "removing $anat_dir"
	fi
fi

# rename fieldmaps to replace magnitude with epi (magnitude as part of filename is a remnant of dcm2niix)
# mv $bids_dir/sub-$subj/ses-00/fmap/sub-${subj}_ses-00_dir-AP_magnitude.json $bids_dir/sub-$subj/ses-00/fmap/sub-${subj}_ses-00_dir-AP_epi.json
# mv $bids_dir/sub-$subj/ses-00/fmap/sub-${subj}_ses-00_dir-AP_magnitude.nii.gz $bids_dir/sub-$subj/ses-00/fmap/sub-${subj}_ses-00_dir-AP_epi.nii.gz
# mv $bids_dir/sub-$subj/ses-00/fmap/sub-${subj}_ses-00_dir-PA_magnitude.json $bids_dir/sub-$subj/ses-00/fmap/sub-${subj}_ses-00_dir-PA_epi.json
# mv $bids_dir/sub-$subj/ses-00/fmap/sub-${subj}_ses-00_dir-PA_magnitude.nii.gz $bids_dir/sub-$subj/ses-00/fmap/sub-${subj}_ses-00_dir-PA_epi.nii.gz

# mv $bids_dir/sub-$subj/ses-01/fmap/sub-${subj}_ses-01_dir-AP_magnitude.json $bids_dir/sub-$subj/ses-01/fmap/sub-${subj}_ses-01_dir-AP_epi.json
# mv $bids_dir/sub-$subj/ses-01/fmap/sub-${subj}_ses-01_dir-AP_magnitude.nii.gz $bids_dir/sub-$subj/ses-01/fmap/sub-${subj}_ses-01_dir-AP_epi.nii.gz
# mv $bids_dir/sub-$subj/ses-01/fmap/sub-${subj}_ses-01_dir-PA_magnitude.json $bids_dir/sub-$subj/ses-01/fmap/sub-${subj}_ses-01_dir-PA_epi.json
# mv $bids_dir/sub-$subj/ses-01/fmap/sub-${subj}_ses-01_dir-PA_magnitude.nii.gz $bids_dir/sub-$subj/ses-01/fmap/sub-${subj}_ses-01_dir-PA_epi.nii.gz

# mv $bids_dir/sub-$subj/ses-02/fmap/sub-${subj}_ses-02_dir-AP_magnitude.json $bids_dir/sub-$subj/ses-02/fmap/sub-${subj}_ses-02_dir-AP_epi.json
# mv $bids_dir/sub-$subj/ses-02/fmap/sub-${subj}_ses-02_dir-AP_magnitude.nii.gz $bids_dir/sub-$subj/ses-02/fmap/sub-${subj}_ses-02_dir-AP_epi.nii.gz
# mv $bids_dir/sub-$subj/ses-02/fmap/sub-${subj}_ses-02_dir-PA_magnitude.json $bids_dir/sub-$subj/ses-02/fmap/sub-${subj}_ses-02_dir-PA_epi.json
# mv $bids_dir/sub-$subj/ses-02/fmap/sub-${subj}_ses-02_dir-PA_magnitude.nii.gz $bids_dir/sub-$subj/ses-02/fmap/sub-${subj}_ses-02_dir-PA_epi.nii.gz

# add "IntendedFor" to fieldmaps
# PRE-SCAN
# beginning='"IntendedFor": ['
# run1="\""ses-00/func/sub-${subj}_ses-00_task-localizer_run-01_bold.nii.gz"\","
# run2="\""ses-00/func/sub-${subj}_ses-00_task-localizer_run-02_bold.nii.gz"\""
# end="],"

# insert="${beginning}${run1} ${run2}${end}"

# sed -i "35 a \ \ ${insert}" $bids_dir/sub-$subj/ses-00/fmap/sub-${subj}_ses-00_dir-AP_epi.json
# sed -i "35 a \ \ ${insert}" $bids_dir/sub-$subj/ses-00/fmap/sub-${subj}_ses-00_dir-PA_epi.json

# # SESSION 1
# beginning='"IntendedFor": ['
# run1="\""ses-01/func/sub-${subj}_ses-01_task-study_run-01_bold.nii.gz"\","
# run2="\""ses-01/func/sub-${subj}_ses-01_task-study_run-02_bold.nii.gz"\","
# run3="\""ses-01/func/sub-${subj}_ses-01_task-study_run-03_bold.nii.gz"\","
# run4="\""ses-01/func/sub-${subj}_ses-01_task-study_run-04_bold.nii.gz"\","
# run5="\""ses-01/func/sub-${subj}_ses-01_task-study_run-05_bold.nii.gz"\","
# run6="\""ses-01/func/sub-${subj}_ses-01_task-study_run-06_bold.nii.gz"\""
# end="],"

# insert="${beginning}${run1} ${run2} ${run3} ${run4} ${run5} ${run6}${end}"

# sed -i "35 a \ \ ${insert}" $bids_dir/sub-$subj/ses-01/fmap/sub-${subj}_ses-01_dir-AP_epi.json
# sed -i "35 a \ \ ${insert}" $bids_dir/sub-$subj/ses-01/fmap/sub-${subj}_ses-01_dir-PA_epi.json

# # SESSION 2
# run1="\""ses-02/func/sub-${subj}_ses-02_task-postscenes_run-01_bold.nii.gz"\","
# run2="\""ses-02/func/sub-${subj}_ses-02_task-familiarization_run-01_bold.nii.gz"\","
# run3="\""ses-02/func/sub-${subj}_ses-02_task-reward_run-01_bold.nii.gz"\","
# run4="\""ses-02/func/sub-${subj}_ses-02_task-reward_run-02_bold.nii.gz"\","
# run5="\""ses-02/func/sub-${subj}_ses-02_task-decision_run-01_bold.nii.gz"\","
# run6="\""ses-02/func/sub-${subj}_ses-02_task-familiarization_run-02_bold.nii.gz"\","
# run7="\""ses-02/func/sub-${subj}_ses-02_task-reward_run-03_bold.nii.gz"\","
# run8="\""ses-02/func/sub-${subj}_ses-02_task-reward_run-04_bold.nii.gz"\","
# run9="\""ses-02/func/sub-${subj}_ses-02_task-decision_run-02_bold.nii.gz"\","
# run10="\""ses-02/func/sub-${subj}_ses-02_task-postfaces_run-01_bold.nii.gz"\""

# insert="${beginning}${run1} ${run2} ${run3} ${run4} ${run5} ${run6} ${run7} ${run8} ${run9} ${run10}${end}"

# sed -i "35 a \ \ ${insert}" $bids_dir/sub-$subj/ses-02/fmap/sub-${subj}_ses-02_dir-AP_epi.json
# sed -i "35 a \ \ ${insert}" $bids_dir/sub-$subj/ses-02/fmap/sub-${subj}_ses-02_dir-PA_epi.json

# # STEP 7 -- run BIDS validator
# echo "Running bids validator"
# /usr/people/rmasis/node_modules/.bin/bids-validator $bids_dir

# #To run bids validator in browser window
# http://bids-standard.github.io/bids-validator/


