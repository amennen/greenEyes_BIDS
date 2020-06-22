# PURPOSE: mask, druncate, compare bold activity
import os
import glob
from shutil import copyfile
import pandas as pd
import json
import numpy as np
from subprocess import call
import time
import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(asctime)s - %(message)s')
import numpy as np
import pickle
import nibabel as nib
import nilearn
from nilearn.image import resample_to_img
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.plotting import show
from nilearn.plotting import plot_roi
from nilearn import image
from nilearn.masking import apply_mask
# get_ipython().magic('matplotlib inline')
import scipy
import matplotlib
import matplotlib.pyplot as plt
from nilearn import image
from nilearn.input_data import NiftiMasker
#from nilearn import plotting
import nibabel
from nilearn.masking import apply_mask
from nilearn.image import load_img
from nilearn.image import new_img_like
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets, svm, metrics
from sklearn.linear_model import Ridge
from sklearn.svm import SVC, LinearSVC
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LeaveOneLabelOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.feature_selection import SelectFwe
from scipy import signal
from scipy.fftpack import fft, fftshift
from scipy import interp
import csv
params = {'legend.fontsize': 'large',
          'figure.figsize': (5, 3),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
font = {'weight': 'bold',
        'size': 22}
plt.rc('font', **font)
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile, f_classif, GenericUnivariateSelect, SelectKBest, chi2
from sklearn.feature_selection import RFE
import os
import seaborn as sns
import pandas as pd
import csv
from scipy import stats
import brainiak
import brainiak.funcalign.srm
import sys
from sklearn.utils import shuffle
import random
from datetime import datetime
random.seed(datetime.now())
from nilearn.image import new_img_like
import scipy.stats as sstats  # type: ignore

def pearsons_mean_corr(A: np.ndarray, B: np.ndarray):
    pearsonsList = []
    if A.shape != B.shape:
        A = flatten_1Ds(A)
        B = flatten_1Ds(B)
    if len(A.shape) == 1:
        A = A.reshape(A.shape[0], 1)
    if len(B.shape) == 1:
        B = B.reshape(B.shape[0], 1)
    assert(A.shape == B.shape)
    num_cols = A.shape[1]
    for col in range(num_cols):
        A_col = A[:, col]
        B_col = B[:, col]
        # ignore NaN values
        nans = np.logical_or(np.isnan(A_col), np.isnan(B_col))
        if np.all(nans == True):  # noqa - np.all needs == comparision not 'is'
            continue
        pearcol = sstats.pearsonr(A_col[~nans], B_col[~nans])
        pearsonsList.append(pearcol)
    pearsons = np.array(pearsonsList)
    if len(pearsons) == 0:
        return np.nan
    pearsons_mean = np.nanmean(pearsons[:, 0])
    return pearsons_mean

bids_dir='/jukebox/norman/amennen/RT_prettymouth/data/bids/Norman/Mennen/5516_greenEyes'
code_dir=bids_dir + '/code'
split_dir=bids_dir + '/derivatives/RTCHECK'
TOM_large = '/jukebox/norman/amennen/prettymouth_fmriprep2/ROI/TOM_large_resampled_maskedbybrain.nii.gz'
offline_path = '/jukebox/norman/amennen/prettymouth_fmriprep2/code/saved_classifiers'

nVox = 2414
nVoxels = nVox
story_TR_1 = 14
story_TR_2 = 464
run_TRs = 450
maskType=1
removeAvg=1
filterType=0 # NEW - as of 4/17
k1=0
k2=25
n_trunc = 12
average_signal_fn = offline_path + '/' +  'averageSignal' + '_' + 'ROI_' + str(maskType) + '_AVGREMOVE_' + str(removeAvg) + '_filter_' + str(filterType) + '_k1_' + str(k1) + '_k2_' + str(k2) + '.npy'
average_signal = np.load(average_signal_fn)
nRuns = 4 # for subject 101 nRuns = 3, for subject 102 nRuns = 4
ses=1

stationsDict = np.load('/jukebox/norman/amennen/prettymouth_fmriprep2/code/upper_right_winners_nofilter.npy').item()
nStations = len(stationsDict)
last_tr_in_station = np.zeros((nStations,))
for st in np.arange(nStations):
  last_tr_in_station[st] = stationsDict[st][-1]

use_spline = 1 # whether or not to use spline at the end moco version
subjects = np.array([101,102])
nSub = len(subjects)
fmriprep_allData = np.zeros((nVoxels,run_TRs,nRuns,nSub))
fmriprep_originalData_all = np.zeros((nVoxels,run_TRs,nRuns,nSub))
fmriprep_zscoredData_all = np.zeros((nVoxels,run_TRs,nRuns,nSub))
fmriprep_removedAvgData_all = np.zeros((nVoxels,run_TRs,nRuns,nSub))

realtime_allData = np.zeros((nVoxels,run_TRs,nRuns,nSub,nStations))
realtime_originalData_all = np.zeros((nVoxels,run_TRs,nRuns,nSub,nStations))
realtime_zscoredData_all = np.zeros((nVoxels,run_TRs,nRuns,nSub,nStations))
realtime_removedAvgData_all = np.zeros((nVoxels,run_TRs,nRuns,nSub,nStations))
realtime_run_mean = np.zeros((nVoxels,nRuns,nSub))
realtime_run_std = np.zeros((nVoxels,nRuns,nSub))
non_brain_vox = {}
for s in np.arange(nSub):
  subjectNum = subjects[s]
  bids_id = 'sub-{0:03d}'.format(subjectNum)
  wf_dir=bids_dir+'/derivatives/work/fmriprep_wf/single_subject_{0:03d}_wf'.format(subjectNum)
  ses_id = 'ses-{0:02d}'.format(ses)

  fmriprep_dir=bids_dir + '/derivatives/fmriprep/{0}/{1}/func'.format(bids_id,ses_id)
  all_fmriprep_files = glob.glob(fmriprep_dir + '/*task-story*space-MNI*preproc_bold.nii.gz')

  func_dir='{0}/{1}/{2}/func/*task-story*.nii.gz'.format(bids_dir,bids_id,ses_id)
  all_func_files = glob.glob(func_dir)
  nruns = len(all_func_files)
  if s == 0:
    nruns = 3
  else:
    nruns = 4
  non_brain_vox[s] = np.array([])
  for r in np.arange(nruns):
    runId='run-{0:02d}'.format(r+1)
    this_fmriprep_file=glob.glob(fmriprep_dir + '/*task-story*' + runId + '*space-MNI*preproc_bold.nii.gz')[0]
    fmriprep_data_masked = apply_mask(this_fmriprep_file,TOM_large)
    fmriprep_masked_data_remove10 = fmriprep_data_masked[10:,:]
    fmriprep_originalData_all[:,:,r,s] = fmriprep_masked_data_remove10[story_TR_1:story_TR_2,:].T

    this_func_file = glob.glob('{0}/{1}/{2}/func/*task-story*{3}*.nii.gz'.format(bids_dir,bids_id,ses_id,runId))[0]
    this_file = this_func_file.split('/')[-1]
    this_file_base = this_file[0:-7]
    split_func_dir='{0}/{1}/{2}/{3}'.format(split_dir,bids_id,ses_id,this_file_base)

    new_tr_files = glob.glob(split_func_dir + '/vol*.nii.gz')
    new_tr_files.sort()
    nTRs = len(new_tr_files)
    all_tr_data_realtime = np.zeros((nTRs,nVox))

    complete_new_nifti_like_image = np.zeros((65,77,49,nTRs))
    for t in np.arange(nTRs):
      this_tr_file = new_tr_files[t].split('/')[-1]
      this_tr_base = this_tr_file[0:-7]
      new_tr_dir = '{0}/{1}'.format(split_func_dir,this_tr_base)
      if use_spline:
        transferred_image = glob.glob(new_tr_dir + '/' + this_tr_base + '_spline_space-MNI.nii.gz')[0]
      else:
        transferred_image = glob.glob(new_tr_dir + '/' + this_tr_base + '_space-MNI.nii.gz')[0] # how to figure out how to specify particularly this one???
      


      t_read_1 = time.time()
      all_tr_data_realtime[t,:] = apply_mask(transferred_image,TOM_large)
      t_read_2 = time.time()
      complete_new_nifti_like_image[:,:,:,t] = nib.load(transferred_image).get_fdata()[:,:,:,0]
      # now compile all data into full TR time course

    realtime_data_nottrunc = new_img_like(transferred_image,complete_new_nifti_like_image)
    realtime_data_masked = apply_mask(realtime_data_nottrunc,TOM_large)
    realtime_masked_data_remove10 = realtime_data_masked[10:,:]
    non_brain_vox_run = np.argwhere(np.std(realtime_data_masked,axis=0) == 0)
    z = np.append(non_brain_vox[s],non_brain_vox_run)
    non_brain_vox[s] = np.unique(z)

    # this takes 1.5-3 seconds to do!
    # epi_masker = NiftiMasker(mask_img = TOM_large,
    #  standardize=False,
    #  t_r=1.5,
    #  memory='nilearn_cache',
    #  memory_level=1,
    #  verbose=0
    # )
    # t_end = last_tr_in_station[st].astype(int)

    # realtime_story_truncated = complete_new_nifti_like_image[:,:,:,n_trunc:]
    # realtime_data_trunc = new_img_like(transferred_image,realtime_story_truncated[:,:,:,0:t_end+story_TR_1-2])
    # t_tr_1 = time.time()
    # realtime_masked_data = epi_masker.fit_transform(realtime_data_trunc)
    # t_tr_2 = time.time()

    fmriprep_originalData_all[:,:,r,s] = stats.zscore(fmriprep_originalData_all[:,:,r,s],axis=1,ddof = 1)
    fmriprep_originalData_all[:,:,r,s] = np.nan_to_num(fmriprep_originalData_all[:,:,r,s])
    fmriprep_zscoredData_all[:,:,r,s] = fmriprep_originalData_all[:,:,r,s]
    fmriprep_removedAvgData_all[:,:,r,s] = fmriprep_originalData_all[:,:,r,s] - average_signal

    for st in np.arange(nStations):
      t_end = last_tr_in_station[st].astype(int)
      realtime_originalData_all[:,0:t_end,r,s,st] = realtime_masked_data_remove10[story_TR_1:story_TR_1+t_end:].T
      realtime_originalData_all[:,0:t_end,r,s,st] = stats.zscore(realtime_originalData_all[:,0:t_end,r,s,st],axis=1,ddof = 1)
      realtime_originalData_all[:,0:t_end,r,s,st] = np.nan_to_num(realtime_originalData_all[:,0:t_end,r,s,st])
      #mean_non_story_data = np.mean(realtime_masked_data_remove10[0:story_TR_1,:],axis=0)
      #std_non_story_data = np.std(realtime_masked_data_remove10[0:story_TR_1,:],axis=0,ddof=1)
      #realtime_originalData_all[:,0:t_end,r,s,st] = np.nan_to_num((realtime_originalData_all[:,0:t_end,r,s,st] - mean_non_story_data[:,np.newaxis])/std_non_story_data[:,np.newaxis])
      realtime_zscoredData_all[:,0:t_end,r,s,st] = realtime_originalData_all[:,0:t_end,r,s,st]
      realtime_removedAvgData_all[:,0:t_end,r,s,st] = realtime_originalData_all[:,0:t_end,r,s,st] - average_signal[:,0:t_end]




# now check each station's classification by using THAT data only
all_z = np.zeros((nSub,4,nStations))
original_z = np.zeros((nSub,4))
# now check pearson correlation
for s in np.arange(nSub):
  if s == 0:
    nruns = 3
  else:
    nruns = 4
  for r in np.arange(nruns):
    for st in np.arange(nStations):
      t_end = last_tr_in_station[st].astype(int)
      # look for ONLY station TRs instead of all
      # now only compare original fmirprep data with that station's data
      this_station_TRs = np.array(stationsDict[st])
      all_z[s,r,st] = pearsons_mean_corr(fmriprep_removedAvgData_all[:,this_station_TRs,r,s],realtime_removedAvgData_all[:,this_station_TRs,r,s,st])


# plot for subject separately
for s in np.arange(nSub):
  if s == 0:
    nruns = 3
  else:
    nruns = 4
  plt.figure(figsize=(10,7))
  for r in np.arange(nruns):
    label = 'run %i' % r
    plt.plot(np.arange(nStations),all_z[s,r,:], label=label)
  title='subject %i' % s
  plt.title(title)
  plt.xlabel('Station number')
  #plt.ylim([0.6,1])
  plt.ylabel('Data correlation')
  plt.legend()
  plt.show()


# classification -- make sure to look and save any bad voxels--don't include in this


nTRs = run_TRs
nRuns = 4
fmriprep_correct_prob = np.zeros((nStations,nRuns,nSub))
realtime_correct_prob = np.zeros((nStations,nRuns,nSub))

# load spatiotemporal pattern and test for each separate station
runOrder = {}
runOrder[0] = ['P', 'P', 'C']
runOrder[1] = ['C', 'C', 'P', 'P']
for s in np.arange(nSub):
    subjectNumber = subjects[s]
    print(subjectNumber)
    if subjectNumber == 101:
        nRuns = 3
    else:
        nRuns = 4
    all_bad_vox_ind = non_brain_vox[s].astype(int)
    for r in np.arange(nRuns):
        for st in np.arange(nStations):
            stationInd = st
            filename = offline_path + '/' + 'UPPERRIGHT_stationInd_' + str(stationInd) + '_' + 'ROI_' + str(maskType) + '_AVGREMOVE_' + str(removeAvg)  + '_filter_' + str(filterType) + '_k1_' + str(k1) + '_k2_' + str(k2)  + '.sav'

            loaded_model = pickle.load(open(filename, 'rb'))
            # test whole spatiotemporal pattern
            this_station_TRs = np.array(stationsDict[stationInd])
            print(this_station_TRs)
            print('***')
            n_station_TRs = len(this_station_TRs)
            fmriprep_testing_data = fmriprep_removedAvgData_all[:,this_station_TRs,r,s]
            fmriprep_testing_data_reshaped = np.reshape(fmriprep_testing_data,(1,nVoxels*n_station_TRs))
            fmriprep_cheating_probability = loaded_model.predict_proba(fmriprep_testing_data_reshaped)[0][1]

            realtime_testing_data = realtime_removedAvgData_all[:,this_station_TRs,r,s,st]
            realtime_testing_data[all_bad_vox_ind,:] = 0
            realtime_testing_data_reshaped = np.reshape(realtime_testing_data,(1,nVoxels*n_station_TRs))
            realtime_cheating_probability = loaded_model.predict_proba(realtime_testing_data_reshaped)[0][1]
            if runOrder[s][r] == 'C':
                fmriprep_correct_prob[st,r,s] = fmriprep_cheating_probability
                realtime_correct_prob[st,r,s] = realtime_cheating_probability
            elif runOrder[s][r] == 'P':
                fmriprep_correct_prob[st,r,s] = 1 - fmriprep_cheating_probability
                realtime_correct_prob[st,r,s] = 1 - realtime_cheating_probability
        print(r,s)




# now check each station's classification by using THAT data only
all_z = np.zeros((nSub,4,nStations))
original_z = np.zeros((nSub,4))
# now check pearson correlation
for s in np.arange(nSub):
  if s == 0:
    nruns = 3
  else:
    nruns = 4
  for r in np.arange(nruns):
    for st in np.arange(nStations):
      t_end = last_tr_in_station[st].astype(int)
      # look for ONLY station TRs instead of all
      # now only compare original fmirprep data with that station's data
      this_station_TRs = np.array(stationsDict[st])
      all_z[s,r,st] = pearsons_mean_corr(fmriprep_removedAvgData_all[:,this_station_TRs,r,s],realtime_removedAvgData_all[:,this_station_TRs,r,s,st])


# plot for subject separately
for s in np.arange(nSub):
  if s == 0:
    nruns = 3
  else:
    nruns = 4
  plt.figure(figsize=(10,7))
  for r in np.arange(nruns):
    label = 'run %i' % r
    plt.plot(np.arange(nStations),all_z[s,r,:], label=label)
  title='subject %i' % s
  plt.title(title)
  plt.xlabel('Station number')
  #plt.ylim([0.6,1])
  plt.ylabel('Data correlation')
  plt.legend()
  plt.show()


# classification -- make sure to look and save any bad voxels--don't include in this


nTRs = run_TRs
nRuns = 4
fmriprep_correct_prob = np.zeros((nStations,nRuns,nSub))
realtime_correct_prob = np.zeros((nStations,nRuns,nSub))

# load spatiotemporal pattern and test for each separate station
runOrder = {}
runOrder[0] = ['P', 'P', 'C']
runOrder[1] = ['C', 'C', 'P', 'P']
for s in np.arange(nSub):
    subjectNumber = subjects[s]
    print(subjectNumber)
    if subjectNumber == 101:
        nRuns = 3
    else:
        nRuns = 4
    all_bad_vox_ind = non_brain_vox[s].astype(int)
    for r in np.arange(nRuns):
        for st in np.arange(nStations):
            stationInd = st
            filename = offline_path + '/' + 'COMBINED_stationInd_' + str(stationInd) + '_' + 'ROI_' + str(maskType) + '_AVGREMOVE_' + str(removeAvg)  + '_filter_' + str(filterType) + '_k1_' + str(k1) + '_k2_' + str(k2)  + '.sav'

            loaded_model = pickle.load(open(filename, 'rb'))
            # test whole spatiotemporal pattern
            this_station_TRs = np.array(stationsDict[stationInd])
            print(this_station_TRs)
            print('***')
            n_station_TRs = len(this_station_TRs)
            fmriprep_testing_data = fmriprep_removedAvgData_all[:,this_station_TRs,r,s]
            fmriprep_testing_data_reshaped = np.reshape(fmriprep_testing_data,(1,nVoxels*n_station_TRs))
            fmriprep_cheating_probability = loaded_model.predict_proba(fmriprep_testing_data_reshaped)[0][1]

            realtime_testing_data = realtime_removedAvgData_all[:,this_station_TRs,r,s,st]
            realtime_testing_data[all_bad_vox_ind,:] = 0
            realtime_testing_data_reshaped = np.reshape(realtime_testing_data,(1,nVoxels*n_station_TRs))
            realtime_cheating_probability = loaded_model.predict_proba(realtime_testing_data_reshaped)[0][1]
            if runOrder[s][r] == 'C':
                fmriprep_correct_prob[st,r,s] = fmriprep_cheating_probability
                realtime_correct_prob[st,r,s] = realtime_cheating_probability
            elif runOrder[s][r] == 'P':
                fmriprep_correct_prob[st,r,s] = 1 - fmriprep_cheating_probability
                realtime_correct_prob[st,r,s] = 1 - realtime_cheating_probability
        print(r,s)


diff_in_pred = fmriprep_correct_prob - realtime_correct_prob
# now plot
# plot for subject separately
for s in np.arange(nSub):
  plt.figure(figsize=(10,7))
  for r in np.arange(nruns):
    label = 'run %i' % r
    plt.plot(np.arange(nStations),diff_in_pred[:,r,s], label=label)
  title='subject %i' % s
  plt.title(title)
  plt.xlabel('Station number')
  #plt.ylim([0.6,1])
  plt.ylabel('Data correlation')
  plt.legend()
  plt.show()

vector_fmriprep = fmriprep_correct_prob.flatten()
vector_realtime = realtime_correct_prob.flatten()
corr_pred = sstats.pearsonr(vector_fmriprep, vector_realtime)


# instead calculate correlation by station
corr_by_station = np.zeros((nStations,))
for st in np.arange(nStations):
  this_station_f = fmriprep_correct_prob[st,:,:].flatten()
  this_station_r = realtime_correct_prob[st,:,:].flatten()
  corr_by_station[st] = sstats.pearsonr(this_station_f,this_station_r)[0]

plt.plot(figsize=(13,10))
plt.plot(np.arange(nStations),corr_by_station, '.', markersize=10)
plt.ylim([0,1.1])
plt.xlabel('station #')
plt.ylabel('fmriprep-rt corr')
plt.title('Total corr = %3.3f' % corr_pred[0])
plt.show()

plt.plot(figsize=(13,10))
plt.plot(vector_fmriprep,vector_realtime, '.', markersize=5)
plt.xlabel('fmriprep prediction')
plt.ylabel('realtime prediction')
plt.title('Correct Category Prediction Probabilities')
print('CORRELATION IS %4.4f' % corr_pred[0])
plt.show()



# NOW FLATTEN AND LOOK AT DIFF
vector_fmriprep = fmriprep_correct_prob.flatten()
vector_realtime = realtime_correct_prob.flatten()

plt.plot(figsize=(13,10))
plt.plot(vector_fmriprep,vector_realtime, '.', markersize=5)
plt.xlabel('fmriprep prediction')
plt.ylabel('realtime prediction')
plt.title('Correct Category Prediction Probabilities')
corr_pred = sstats.pearsonr(vector_fmriprep, vector_realtime)
plt.show()

stationsDict
m_tr_station = np.zeros((nStations,))
for i in np.arange(nStations):
    st = i
    m_tr_station[i] = np.mean(stationsDict[st])

sns.set(style="white")
sns.set(font_scale=2)
plt.figure(figsize=(10,7))
g1 = '#18a377'
g2 = '#88ab6f'
p1 = '#b02543'
p2 = '#b36274'
colors = [p1, p2, g1]
plt.plot(figsize=(10,7))
for r in np.arange(3):
    sns.lineplot(x=m_tr_station,y=correct_prob[:,r,0],lw=3,color=colors[r],marker=".",markerSize=15,mec='k',mfc='k')
plt.legend(('P1', 'P2', 'C1', 'ERR'),loc=3)
plt.xlabel('TR #')
plt.ylabel('Probability(correct)')
plt.title('Subject 1')
plt.ylim([0,1])

plt.figure(figsize=(10,7))
colors = [g1, g2, p1, p2]
plt.plot(figsize=(10,7))
for r in np.arange(4):
    sns.lineplot(x=m_tr_station,y=correct_prob[:,r,1],lw=3,color=colors[r],marker=".",markerSize=15,mec='k',mfc='k')
plt.legend(('C1', 'C2', 'P1', 'P2'),loc=3)
plt.xlabel('TR #')
plt.ylabel('Probability(correct)')
plt.title('Subject 2')
plt.ylim([0,1])

#np.mean(cheating_prob[:,0:2,0])
print(np.mean(correct_prob[:,0:3,0]))
print(np.mean(correct_prob[:,:,1]))