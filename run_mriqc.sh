#!/bin/bash

data_dir=/jukebox/norman/amennen/RT_prettymouth/data #this is my study directory
bids_dir=$data_dir/bids/Norman/Mennen/5516_greenEyes #this is where BIDS formatted data will end up and should match the program card on the scanner


singularity run --cleanenv \
    --bind $bids_dir:/home \
    /jukebox/hasson/singularity/mriqc/mriqc-v0.10.4.sqsh \
    --participant-label sub-$1 \
    --correct-slice-timing --no-sub \
    --nprocs 8 -w /home/derivatives/work \
    /home /home/derivatives/mriqc participant
