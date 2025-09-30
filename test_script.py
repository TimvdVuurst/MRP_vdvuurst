import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("-T","--mass_range" type= tuple, default=(13,13.5))
parser.add_argument("-T","--mass_range", type = float, nargs = '+', default=[13,13.5])
args = parser.parse_args()

print(args.mass_range, type(args.mass_range))