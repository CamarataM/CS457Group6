# Taken From 04/16/2023 12:11 PM: https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/
# Perceptron Algorithm on the Sonar Dataset
from math import sqrt
import random
from typing import List

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = random.randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)

	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1

	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	# Create a single initial fold with just the dataset.
	folds = [dataset]

	if use_folds:
		folds = cross_validation_split(dataset, n_folds)

	scores = list()
	for fold in folds:
		train_set = list(folds)

		if use_folds:
			train_set.remove(fold)

		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]

		# print(str(predicted))
		# print(str(actual))

		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)

	return scores

# Make a prediction with weights
def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]

	# print(str(activation))

	return 1.0 if activation >= 0.0 else 0.0

# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		for row in train:
			prediction = predict(row, weights)
			error = row[-1] - prediction
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]

	return weights

# Perceptron Algorithm With Stochastic Gradient Descent
def perceptron(train, test, l_rate, n_epoch):
	predictions = list()
	weights = train_weights(train, l_rate, n_epoch)

	if print_weights:
		# Print the weights of the current Perceptron iteration.
		print("Weights: " + str(weights))

	for row in test:
		prediction = predict(row, weights)
		predictions.append(prediction)

	return(predictions)

# The bottom line of the 'L' is called an 'arm' or 'horizontal bar' depending on where you look, see https://en.wikipedia.org/wiki/Typeface_anatomy and https://graphicdesign.stackexchange.com/questions/156812/typography-what-is-the-bottom-end-of-the-l-called and https://english.stackexchange.com/questions/583162/words-for-the-long-and-short-parts-of-the-letter-l

# Creates a linear array which contains either an 'I' or an 'L' depending on if 'l_arms' is greater than 0. 'right_shift' will shift the starting point for the 'I' or 'L' either left for negative values or right for positive values.
def generate_matrix(width : int, character_height = -1, l_arms = 0, down_shift = 0, right_shift = 0):
	matrix_list_length = (width * width) + 1
	matrix = [0] * matrix_list_length

	# The minimum character height is 2, so it lock it to values equal to or greater than that.
	character_height = max(character_height, 2)

	# Calculate the character starting index, which is equal to the amount of positions to shift to the right plus the amount of down shifts times the width of the matrix.
	starting_index = right_shift + (down_shift * width)

	# The first index we need to put a one is equal to the starting index, meaning we will skip over all elements until we reach it.
	next_1_index = starting_index

	for i in range(matrix_list_length):
		# If we still have characters to calculate, check if the index is the next one index. If it is, set it to one and decrement the character height by one.
		if character_height > 0:
			if next_1_index <= 0:
				matrix[i] = 1
				next_1_index = width
				character_height -= 1

			next_1_index -= 1

	# Reverse the dataset array, then for every one while we still have l_arms to place, set the next element in the array equal to one.
	for i in reversed(range(matrix_list_length - 1)):
		if l_arms <= 0:
			break

		if matrix[i] == 1:
			matrix[i + 1] = 1
			l_arms -= 1

	# Set the final value of the dataset equal to the constant value.
	matrix[len(matrix) - 1] = FIXED_VALUE

	return matrix

# Prints a matrix array in a more human-readable form.
def print_array_matrix_form(matrix : List[int]):
	current_string = ""
	wrap_modulus = int(sqrt(len(matrix) - 1))
	for i in range(len(matrix)):
		if i != 0 and i % wrap_modulus == 0:
			print(current_string)
			current_string = ""

		current_string += str(matrix[i]) + " "

	print(current_string)

random.seed(1)

# Controls whether folds should be used at all, or if the Perceptron should just be trained and ran on the same dataset.
use_folds = False

# Determines how many pools the dataset is broken into, with the first being considered the "training" dataset. Should be greater than or equal to 2.
number_of_folds = 2

# Determines how fast the Perceptron will adjust the weights.
learning_rate = 0.01

# Controls how many times the Perceptron should be ran over a specific dataset during training. More epoch iterations will result in more over-fitting.
number_of_epochs = 1

# Controls the diameter (width and height) of the dataset matrices.
matrix_width = 3

# Controls if for random dataset generation if 'L' matrices should be checked for being valid or not. Since this is mathematically controlled, this is not technically necessary, but ensures that any strange behavior is caught.
do_verify_l_shape = True

# Controls whether dataset order is shuffled or not.
shuffle_dataset = True

# Controls whether the random matrices should have random down shifts.
random_down_shifts = True

# Controls whether the random matrices should have random right shifts.
random_right_shifts = True

# Controls whether the random matrices should have random character heights.
random_character_heights = True

print_weights = False

# Set which value corresponds to which matrices for the truth values.
I_MATRIX_TRUTH_VALUE = 0
L_MATRIX_TRUTH_VALUE = 1
FIXED_VALUE = -1

def verify_l_shape(matrix : List[int], expected_l_arms : int):
	counted_l_arms = 0
	for i in range(len(matrix)):
		# Count the amount of 'L' arms the matrix list has. If this is equal to 0, then we have a malformed dataset array for the 'L' shape.
		if matrix[i] == 1 and i + 1 < len(matrix) and matrix[i + 1] == 1:
			counted_l_arms += 1

	# If the counted 'L' arms is not equal to the expected generated value, print an error and exit the program.
	if counted_l_arms != expected_l_arms:
		raise Exception("Invalid 'L' dataset array, counted 'L' arms not equal to generated value (Counted: " + str(counted_l_arms) + ", Expected: " + str(expected_l_arms) + "). Dataset: " + str(matrix))

# Generates a dataset with a set number of 'I' and 'L' matrices with random shifts and heights.
def generate_dataset(i_matrix_count : int, l_matrix_count : int):
	dataset = []

	for _ in range(i_matrix_count + l_matrix_count):
		# Set the amount of arms initially to 1 to generate an 'L', else if i_matrix_count is greater than 0, set it to 0 to generate an 'I' matrix.
		l_arms = 1

		# If we still have 'I' matrices to produce, set the amount of arms to zero and decrement the i_matrix_count
		if i_matrix_count > 0:
			l_arms = 0
			i_matrix_count -= 1
		else:
			# Not strictly necessary, more added for completion.
			l_matrix_count -= 1

		is_l = l_arms > 0

		character_height = matrix_width

		if random_character_heights:
			# Generate a random character height between 2 (the minimum) and the total width of the matrix.
			character_height = random.randrange(2, matrix_width)

		down_shift = 0
		right_shift = 0

		maximum_down_shift = matrix_width - character_height
		maximum_right_shift = matrix_width

		# If the matrix is an 'L' shape, the maximum shift we can do is the width of the matrix - 1, as anything more will cut the arm off.
		if is_l:
			maximum_right_shift = matrix_width - 1

		if random_down_shifts:
			# The maximum amount of positions we can shift down is equal to the height of the matrix minus the height of the character.
			down_shift = random.randrange(0, maximum_down_shift)

		if random_right_shifts:
			right_shift = random.randrange(0, maximum_right_shift)

		# Generate a new dataset with a random shift between 0 and the max shift variable.
		new_dataset_matrix = generate_matrix(width=matrix_width, character_height=character_height, l_arms=l_arms, down_shift=down_shift, right_shift=right_shift)

		# Verify the shape if required.
		if do_verify_l_shape and is_l:
			verify_l_shape(new_dataset_matrix, l_arms)

		# Append to the end whether the resulting dataset is an 'I' or an 'L', required for the Perceptron to check whether the produced output was correct or not.
		new_dataset_matrix.append(L_MATRIX_TRUTH_VALUE if is_l else I_MATRIX_TRUTH_VALUE)

		# Append the new dataset to the dataset list.
		dataset.append(new_dataset_matrix)

	return dataset

# Generate a dataset with a random number of 'I' and 'L' matrices.
def generate_random_dataset(dataset_size : int):
	i_matrix_count = random.randint(0, dataset_size)
	return generate_dataset(i_matrix_count=i_matrix_count, l_matrix_count=dataset_size - i_matrix_count)

# Tests a specific dataset.
def test_dataset(dataset : List[List[int]]):
	# Shuffle the dataset if required.
	if shuffle_dataset:
		random.shuffle(dataset)

	if number_of_folds > len(dataset):
		print("Reduced number of folds to be equal to the amount of elements in dataset, which is " + str(len(dataset)))

	# Evaluate the Perceptron algorithm passing in the generated dataset and the various parameters related to the Perceptron. Take the minimal value between the size of the dataset and the number of d folds we are attempting to have to avoid an exception.
	scores = evaluate_algorithm(dataset, perceptron, min(len(dataset), number_of_folds), learning_rate, number_of_epochs)
	accuracy = sum(scores) / float(len(scores))

	return (accuracy, scores)

# Generates a dataset of randomly generated matrices, with (pseudo-)equal chances of being an 'I' or 'L'. The parameter passed is how many matrices to generate for the dataset.
# dataset = generate_random_dataset(20)

# Generates a dataset which contains every possible 'I' and 'L'
# dataset = generate_minimal_complete_dataset()

# Generates a dataset with a fixed number of 'I' and 'L' matrices with random shift values (including duplicates).
# dataset = generate_dataset(12, 12)

dataset = None

# Contains a list of tuples representing how many 'I' and 'L' matrices to produce with random shifts. Each will be tested with the accuracy being printed to the console.
dataset_test_cases = [
	(1, 1),
	(5, 5),
	(10, 10),
	(1, 100),
	(10, 100),
	(12, 12),
	(25, 25),
	(50, 50),
	(75, 75),
	(76, 76),
	(100, 100),
	(250, 250),
	(500, 500),
	(1000, 1000),
	(5000, 5000)
]

# Contains a list of tuples representing how many 'I' and 'L' matrices to produce with random shifts to test with the learning rate test cases.
learning_rate_dataset_test_cases = [
	(1, 1),
	(10, 10),
	(25, 25),
	(25, 75),
	(50, 50),
	(100, 100),
]

# Contains a list of learning rates to test.
learning_rate_test_cases = [
	0.00001,
	0.1,
	0.25,
	0.5,
	0.75,
	1
]

if dataset != None and len(dataset) > 0:
	accuracy, scores = test_dataset(dataset)
	print('Scores: %s' % scores)
	print('Mean Accuracy: %.3f%%' % (accuracy))
	print("")

print("Learning Rate: " + str(learning_rate))
print("Matrix Size: " + str(matrix_width) + " x " + str(matrix_width))
if dataset_test_cases != None and len(dataset_test_cases) > 0:
	for dataset_test_case in dataset_test_cases:
		dataset = generate_dataset(dataset_test_case[0], dataset_test_case[1])
		accuracy, scores = test_dataset(dataset)

		print(str(dataset_test_case[0]) + " I, " + str(dataset_test_case[1]) + " L. Accuracy: %.3f%%" % (accuracy))

print("")

if learning_rate_dataset_test_cases != None and len(learning_rate_dataset_test_cases) > 0:
	original_learning_rate = learning_rate

	for learning_rate_test_case in learning_rate_test_cases:
		learning_rate = learning_rate_test_case
		print("Learning Rate: " + str(learning_rate))
		for dataset_test_case in learning_rate_dataset_test_cases:
			dataset = generate_dataset(dataset_test_case[0], dataset_test_case[1])
			accuracy, scores = test_dataset(dataset)

			print(str(dataset_test_case[0]) + " I, " + str(dataset_test_case[1]) + " L. Accuracy: %.3f%%" % (accuracy))

		if learning_rate != learning_rate_test_cases[-1]:
			print("")

	learning_rate = original_learning_rate
