import numpy as np
import argparse
import os
import csv

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--test', action='store_true')
	return parser.parse_args()

def main():
	args = parse_args()
	min_value = None
	min_file = None
	max_value = None
	max_file = None
	type_ = 'train' if not args.test else 'test'
	for filename in os.listdir('.'):
		if type_  not in filename or 'max_min' in filename:
			continue
		with open(filename) as f:
			value = np.mean([float(line[0]) for line in list(csv.reader(f))[-3:]])
			if min_file is None or min(min_value, value) == value:
				min_file = filename
				min_value = value
			if max_file is None or max(max_value, value) == value:
				max_file = filename
				max_value = value
	with open("max_min_{}".format(type_), "w") as f:
		f.write("min: ")
		f.write(min_file + "\n")
		f.write("max: ")
		f.write(max_file + "\n")

if __name__ == "__main__":
    main()