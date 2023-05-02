import math
import os
from pathlib import Path
import random
from typing import List
# TODO: Could use the 'bigfloat' library to improve floating-point precision, see https://pypi.org/project/bigfloat/

# Fix the seed of random to make it deterministic.
random.seed(0)

# Inclusive
x_lower_range = 1
# Inclusive
x_upper_range = 100

# Inclusive
y_lower_range = 1
# Inclusive
y_upper_range = 100

# Whether to append a fixed value to the dataset.
use_fixed_value = False
fixed_value = -1

# Whether to force the inclusion of the minimum and maximum values for x and y.
force_minimum_and_maximum_value = True

dataset_folder_path = "datasets"

# TODO: Switch to dict() which supports key file paths and value dataset sizes. 
dataset_output_file_path_list = [dataset_folder_path + os.sep + "training.arff", dataset_folder_path + os.sep + "test.arff"]
dataset_sizes = [10, 100, 1000, 10000]

generate_random_range = False
generate_uniform_range = True

def squared(number):
	return math.pow(number, 2)

def calculate_function(x : float, y : float):
	sin_part = math.sin(math.pi * 10 * x + 10/(1 + squared(y)))
	
	# math.log is natural logarithm, see https://docs.python.org/3/library/math.html#math.log
	ln_part = math.log(squared(x) + squared(y))
	
	return sin_part + ln_part

def random_x():
	return random.uniform(x_lower_range, x_upper_range)

def random_y():
	return random.uniform(y_lower_range, y_upper_range)

def is_inclusive(lower, upper):
	# Check that if random.random() produces 1 (maximum value) that the lower value minus the upper value is equal to the upper value.
	# Taken From 04/27/2023 10:52 AM: https://docs.python.org/3/library/random.html#random.uniform
	return lower + (upper-lower) * 1 == upper

# Check that that distance between the two values is less than the epsilon.
def within_epsilon(valueOne, valueTwo, epsilon = 0.000000001):
	return abs(valueOne - valueTwo) < epsilon

RELATION = "@RELATION"
ATTRIBUTE = "@ATTRIBUTE"
NUMERIC = "NUMERIC"
DATA = "@DATA"

def generate_dataset_line(x : float, y : float):
	output = calculate_function(x, y)

	output_string = str(x) + "," + str(y) + ","

	if use_fixed_value:
		output_string += str(fixed_value) + ","

	output_string += str(output)

	return output_string

# ARFF Format Taken From 04/27/2023 11:13 AM: https://www.cs.waikato.ac.nz/ml/weka/arff.html
def generate_dataset(output_file_path : str, dataset_size : int):
	with open(output_file_path, 'w') as file:
		def write(line = ""):
			file.write(str(line) + '\n')

		write("% 1. Title: sin ln Function Dataset")
		write("%")
		write("% 2. Sources:")
		write("%      (a) Creators: Michael Camarata, Ariesh Grover, Miguel Lima, Jaden Martinez")
		write("%      (b) Donar: N/A")
		write("%      (c) Date: April, 2023")
		write(RELATION + " math")

		write()

		# TODO: Switch to dict(), linking string attributes to string types instead of assuming NUMERIC.
		attributes = ["x", "y"]

		if use_fixed_value:
			attributes.append("fixed_value")

		attributes.append("output")

		# Grab the maximum padding for the longest string, then left-pad every string with spaces to that length.
		padding = 0
		for attribute in attributes:
			padding = max(padding, len(attribute))

		for i in range(len(attributes)):
			attributes[i] = attributes[i].ljust(padding)

		for attribute in attributes:
			write(ATTRIBUTE + " " + attribute + "  " + NUMERIC)

		write()

		write(DATA)

		dataset_lines : List[str] = []
		if generate_random_range and force_minimum_and_maximum_value:
			dataset_lines.append(generate_dataset_line(x_lower_range, y_lower_range))
			dataset_lines.append(generate_dataset_line(x_upper_range, y_upper_range))

		x = 0
		y = 0

		# If we are set to generate a uniform range, calculate the uniform offset by dividing the maximum range value by the dataset size.
		x_y_offset = 0
		if not generate_random_range and generate_uniform_range:
			assert x_upper_range == y_upper_range

			x_y_offset = max(x_upper_range, y_upper_range) / dataset_size

		# Generate random data lines to fill the rest of the dataset size.
		for _ in range(dataset_size - len(dataset_lines)):
			if generate_random_range:
				x = random_x()
				y = random_y()

			x += x_y_offset
			y += x_y_offset

			dataset_lines.append(generate_dataset_line(x, y))

		assert len(dataset_lines) == dataset_size

		# Shuffle the dataset lines.
		random.shuffle(dataset_lines)

		for dataset_line in dataset_lines:
			write(dataset_line)

def main():
	# Ensure that the lower and upper range of x and y can produce the upper value (inclusive).
	assert is_inclusive(x_lower_range, x_upper_range)
	assert is_inclusive(y_lower_range, y_upper_range)

	# Validated by plugging sin(pi * 10 * 84 + 10/(1 + 76^2)) + ln(84^2 + 76^2) into Demos ( https://www.desmos.com/calculator )
	# Check that the calculate value distance from the expected value is less than 0.000000001.
	# TODO: Should likely automate this process rather than having a fixed, limited test set. Could use eval(), but this option was disregarded due to the dangers of eval() (see https://stackoverflow.com/a/9558001 ). One alternative would be numexpr ( https://github.com/pydata/numexpr ), although it is likely overkill for this application, so a "safe" eval alternative would probably be better.
	assert within_epsilon(calculate_function(84, 76), 9.46142833149)

	# Create the required output folder structure.
	Path(dataset_folder_path).mkdir(parents=True, exist_ok=True)

	for dataset_output_file_path in dataset_output_file_path_list:
		for dataset_size in dataset_sizes:
			output_file_path = dataset_output_file_path.replace(".arff", "_" + str(dataset_size) + ".arff")

			if use_fixed_value:
				output_file_path = output_file_path.replace(".arff", "_fixed.arff")

			if generate_random_range:
				output_file_path = output_file_path.replace(".arff", "_random.arff")

			if generate_uniform_range:
				output_file_path = output_file_path.replace(".arff", "_uniform.arff")

			generate_dataset(output_file_path, dataset_size)

if __name__ == "__main__":
	main()
