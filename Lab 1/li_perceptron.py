# Taken From 04/16/2023 12:11 PM: https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/
# Perceptron Algorithm on the Sonar Dataset
from math import sqrt
from random import seed
from random import randrange
from csv import reader

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
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
	for row in test:
		prediction = predict(row, weights)
		predictions.append(prediction)
	return(predictions)

# Creates a linear array which contains either an 'I' or an 'L' depending on if 'l_tails' is greater than 0. 'left_shift' will shift the starting point for the 'I' or 'L' either left for negative values or right for positive values.
def generate_dataset(width, l_tails = 0, left_shift = 0):
	dataset_max = (width * width) + 1
	dataset = [0] * dataset_max

	next_1_index = left_shift % width
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

def print_dataset_matrix_form(dataset):
	current_string = ""
	wrap_modulus = int(sqrt(len(dataset) - 1))
	for i in range(len(dataset)):
		if i != 0 and i % wrap_modulus == 0:
			print(current_string)
			current_string = ""

		current_string += str(dataset[i]) + " "

	print(current_string)

# Test the Perceptron algorithm on the sonar dataset
# seed(1)
# # load and prepare data
# filename = 'sonar.all-data.csv'
# dataset = load_csv(filename)
# for i in range(len(dataset[0]) - 1):
# 	str_column_to_float(dataset, i)
# # convert string class to integers
# str_column_to_int(dataset, len(dataset[0])-1)
# # evaluate algorithm
# n_folds = 3
# l_rate = 0.01
# n_epoch = 500
# scores = evaluate_algorithm(dataset, perceptron, n_folds, l_rate, n_epoch)
# print('Scores: %s' % scores)
# print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

# print(str(generate_dataset(5, 2)))
print_dataset_matrix_form(generate_dataset(5, 1, 0))