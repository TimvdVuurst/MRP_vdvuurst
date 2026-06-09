#!/bin/bash
echo "Initializing, running initial conditions..."

python3 ONEHALO_initial_conditions.py -SB 1 -D 1 -F 0 -RR 1 -FF sigma1_poly4

echo "Initial conditions completed, proceeding to fit..."

python3 ./batchjobs/onehalo_MADD_batchjob.py -SF 1 -NC 30 -RM 1 -O 1 -FF sigma1_poly4

echo "Completed!"