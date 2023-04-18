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
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)

	return scores

# Make a prediction with weights
def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]

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

	# Print the weights at the current Perceptron iteration.
	print("Weights: " + str(weights))

	for row in test:
		prediction = predict(row, weights)
		predictions.append(prediction)

	return(predictions)

# Creates a linear array which contains either an 'I' or an 'L' depending on if 'l_tails' is greater than 0. 'right_shift' will shift the starting point for the 'I' or 'L' either left for negative values or right for positive values.
def generate_matrix(width : int, l_tails = 0, right_shift = 0):
	dataset_max = (width * width) + 1
	dataset = [0] * dataset_max

	next_1_index = right_shift % width
	for i in range(dataset_max):
		if next_1_index == 0:
			dataset[i] = 1
			next_1_index = width

		next_1_index -= 1

	for i in reversed(range(dataset_max - 1)):
		if l_tails <= 0:
			break

		if dataset[i] == 1:
			dataset[i + 1] = 1
			l_tails -= 1

	dataset[len(dataset) - 1] = -1

	return dataset

# Prints a dataset matrix in a more human-readable form.
def print_dataset_matrix_form(dataset : List[int]):
	current_string = ""
	wrap_modulus = int(sqrt(len(dataset) - 1))
	for i in range(len(dataset)):
		if i != 0 and i % wrap_modulus == 0:
			print(current_string)
			current_string = ""

		current_string += str(dataset[i]) + " "

	print(current_string)

random.seed(1)

# Produced the greatest result with just plugging in random numbers for a bit. Anything less than 50 will produce about 80% accuracy.
number_of_folds = 100

learning_rate = 0.01

# TODO: Describe what this actually controls, not sure yet.
number_of_epochs = 500

# Controls the diameter (width and height) of the dataset matrices.
matrix_width = 5

# Controls if for random dataset generation if 'L' matrices should be checked for being valid or not. Since this is mathematically controlled, this is not technically necessary, but ensures that any strange behaviour is caught.
verify_l_shape = True

# Controls whether dataset order is shuffled or not.
shuffle_dataset = True

# Set which value corresponds to which matrices for the truth values.
I_MATRIX_TRUTH_VALUE = 0
L_MATRIX_TRUTH_VALUE = 1

def generate_random_dataset(dataset_size : int):
	dataset = []

	for _ in range(dataset_size):
		# Will randomly generate whether the dataset produced will be an 'I' or an 'L' by setting the amount of tails for the matrix to '0' or '1' respectively.
		l_tails = random.randrange(0, 2)
		is_l = l_tails > 0

		max_shift = matrix_width

		# If the matrix is an 'L' shape, the maximum shift we can do is the width of the matrix - 1, as anything more will cut the tail off.
		if is_l:
			max_shift = matrix_width - 1

		# Generate a new dataset with a random shift between 0 and the max shift variable.
		new_dataset_matrix = generate_matrix(matrix_width, l_tails, random.randrange(0, max_shift))

		if verify_l_shape and is_l:
			counted_l_tails = 0
			for i in range(len(new_dataset_matrix)):
				# Count the amount of 'L' tails the matrix list has. If this is equal to 0, then we have a malformed dataset array for the 'L' shape.
				if new_dataset_matrix[i] == 1 and i + 1 < len(new_dataset_matrix) and new_dataset_matrix[i + 1] == 1:
					counted_l_tails += 1

			# If the counted 'L' tails is not equal to the expected generated value, print an error and exit the program.
			if counted_l_tails != l_tails:
				raise Exception("Invalid 'L' dataset array, counted 'L' tails not equal to generated value (Counted: " + str(counted_l_tails) + ", Expected: " + str(l_tails) + "). Dataset: " + str(new_dataset_matrix))

		# Append to the end whether the resulting dataset is an 'I' or an 'L', required for the Perceptron to check whether the produced output was correct or not.
		new_dataset_matrix.append(L_MATRIX_TRUTH_VALUE if is_l else I_MATRIX_TRUTH_VALUE)

		# Append the new dataset to the dataset list.
		dataset.append(new_dataset_matrix)

	return dataset

def generate_minimal_complete_dataset():
	dataset = []

	# Generate all possible 'I' values, which is all 'I' shifted to the right 0 through matrix width amount of times.
	for right_shift in range(matrix_width):
		new_dataset_matrix = generate_matrix(matrix_width, 0, right_shift)

		# Append a 0 to indicate that this is an 'I' matrix.
		new_dataset_matrix.append(I_MATRIX_TRUTH_VALUE)

		# Append the new dataset to the dataset list.
		dataset.append(new_dataset_matrix)

	# Generate all possible 'L' values, which is all 'L' shifted to the right 0 through matrix width minus one amount of times.
	for right_shift in range(matrix_width - 1):
		new_dataset_matrix = generate_matrix(matrix_width, 1, right_shift)

		# Append a 1 to indicate that this is an 'L' matrix.
		new_dataset_matrix.append(L_MATRIX_TRUTH_VALUE)
		
		# Append the new dataset to the dataset list.
		dataset.append(new_dataset_matrix)

	return dataset

# Generates a dataset of randomly generated matrices, with (pseudo-)equal chances of being an 'I' or 'L'. The parameter passed is how many matrices to generate for the dataset.
dataset = generate_random_dataset(100)

# Generates a dataset which contains every possible 'I' and 'L'
# dataset = generate_minimal_complete_dataset()

if shuffle_dataset:
	random.shuffle(dataset)

if number_of_folds > len(dataset):
	number_of_folds = len(dataset)
	print("Reduced number of folds to be equal to the amount of elements in dataset, which is " + str(number_of_folds))

# Evaluate the Perceptron algorithm passing in the generated dataset and the various parameters related to the Perceptron.
scores = evaluate_algorithm(dataset, perceptron, number_of_folds, learning_rate, number_of_epochs)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
