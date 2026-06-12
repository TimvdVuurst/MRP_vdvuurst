#!/bin/bash
echo "Initializing, running initial conditions..."

python3 ONEHALO_initial_conditions.py -SB 1 -D 1 -F 0 -RR 0 -FF sigma2_poly2

echo "Initial conditions completed, proceeding to fit..."

python3 ./batchjobs/onehalo_MADD_batchjob.py -SF 1 -NC 30 -RM 1 -O 1 -FF sigma2_poly2

echo "Completed!"