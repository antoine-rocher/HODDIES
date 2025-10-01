#!/bin/bash


source /global/common/software/desi/users/adematti/cosmodesi_environment.sh 2025_03

for halodir in 27 29 30 31 32 34 35 36 37 38 39 40 41
do 
    for i in {0..9}
    do 
        srun -n 1 --cpus-per-task=32 python /global/homes/a/arocher/Code/postdoc/HOD/Dev/HODDIES/HODDIES/desi/Uchuu/preprocees_Uchuu.py --batch_num $i --halodir $halodir &
    done
done