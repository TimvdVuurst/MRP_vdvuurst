#!/bin/bash
echo "Initializing, running intial conditions..."

python3 ONEHALO_initial_conditions.py -SB 1 -RR 0

echo "Initial conditions completed, proceeding to fit..."

python3 ./batchjobs/onehalo_joint_batchjob.py -SF 1 -NC 30 -RM 1

echo "Completed!"