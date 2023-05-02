import math
import os
import random
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

# TODO: Switch to dict() which supports key file paths and value dataset sizes. 
dataset_output_file_path_list = ["datasets" + os.sep + "training.arff", "datasets" + os.sep + "test.arff"]
dataset_sizes = [10, 100, 1000, 10000]

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

		for _ in range(dataset_size):
			x = random_x()
			y = random_y()

			output = calculate_function(x, y)

			output_string = str(x) + "," + str(y) + ","

			if use_fixed_value:
				output_string += str(fixed_value) + ","

			output_string += str(output)

			write(output_string)

def main():
	# Ensure that the lower and upper range of x and y can produce the upper value (inclusive).
	assert is_inclusive(x_lower_range, x_upper_range)
	assert is_inclusive(y_lower_range, y_upper_range)

	# Validated by plugging sin(pi * 10 * 84 + 10/(1 + 76^2)) + ln(84^2 + 76^2) into Demos ( https://www.desmos.com/calculator )
	# Check that the calculate value distance from the expected value is less than 0.000000001.
	# TODO: Should likely automate this process rather than having a fixed, limited test set. Could use eval(), but this option was disregarded due to the dangers of eval() (see https://stackoverflow.com/a/9558001 ). One alternative would be numexpr ( https://github.com/pydata/numexpr ), although it is likely overkill for this application, so a "safe" eval alternative would probably be better.
	assert within_epsilon(calculate_function(84, 76), 9.46142833149)

	for dataset_output_file_path in dataset_output_file_path_list:
		for dataset_size in dataset_sizes:
			output_file_path = dataset_output_file_path.replace(".arff", "_" + str(dataset_size) + ".arff")

			if use_fixed_value:
				output_file_path = output_file_path.replace(".arff", "_fixed.arff")

			generate_dataset(output_file_path, dataset_size)

if __name__ == "__main__":
	main()
