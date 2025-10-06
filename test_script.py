import argparse
parser = argparse.ArgumentParser()
# parser.add_argument('-M1','--lower_mass_range', type = float, default = 12, help = 'Loewr bound of the mass range. This is inclusive!')
# parser.add_argument('-M2','--upper_mass_range', type = float, default = 16, help = 'Upper bound of the mass range. This is exclusive!') #always EXCLUSIVE upper bound
parser.add_argument('--o', type = int, default = 1)
args = parser.parse_args()
overwrite = bool(args.o)

print(args.o, overwrite)