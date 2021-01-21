
import tensorflow as tf
import numpy as np
import random
from utils import *



# Function to get word2vec representations
#
# Arguments:
# reviews: A list of strings, each string represents a review
#
# Returns: mat (numpy.ndarray) of size (len(reviews), dim)
# mat is a two-dimensional numpy array containing vector representation for ith review (in input list reviews) in ith row
# dim represents the dimensions of word vectors, here dim = 300 for Google News pre-trained vectors
def w2v_rep(reviews):

	dim = 300
	mat = np.zeros((len(reviews), dim))
	pretrained_data = load_w2v()
	review=0
	while review < len(reviews):
		avg = 0
		tokens_list = get_tokens(reviews[review])
		for token in tokens_list:
			try:
				data_to_load = pretrained_data[token]
				mat[review] = np.add(mat[review],data_to_load)
				avg+=1
			except:
				continue
		try:
			mat[review] = np.divide(mat[review],avg)
		except:
			continue
		review+=1
	return mat


# Function to build a feed-forward neural network using tf.keras.Sequential model. You should build the sequential model
# by stacking up dense layers such that each hidden layer has 'relu' activation. Add an output dense layer in the end
# containing 1 unit, with 'sigmoid' activation, this is to ensure that we get label probability as output
#
# Arguments:
# params (dict): A dictionary containing the following parameter data:
#					layers (int): Number of dense layers in the neural network
#					units (int): Number of units in each dense layer
#					loss (string): The type of loss to optimize ('binary_crossentropy' or 'mse)
#					optimizer (string): The type of optimizer to use while training ('sgd' or 'adam')
#
# Returns:
# model (tf.keras.Sequential), a compiled model created using the specified parameters
def build_nn(params):
	model = tf.keras.Sequential()
	num_layers=params['layers']
	num_units=params['units']
	loss_type=params['loss']
	optimizer_type=params['optimizer']
	model.add(tf.keras.layers.Dense(num_units, input_dim=300, activation='relu'))
	i=0
	while i < num_layers-1:
		model.add(tf.keras.layers.Dense(num_units, activation='relu'))
		i+=1
	model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
	model.compile(loss=loss_type, optimizer=optimizer_type)

	return model


# Function to select the best parameter combination based on accuracy by evaluating all parameter combinations
# This function should train on the training set (X_train, y_train) and evluate using the validation set (X_val, y_val)
#
# Arguments:
# params (dict): A dictionary containing parameter combinations to try:
#					layers (list of int): Each element specifies number of dense layers in the neural network
#					units (list of int): Each element specifies the number of units in each dense layer
#					loss (list of string): Each element specifies the type of loss to optimize ('binary_crossentropy' or 'mse)
#					optimizer (list of string): Each element specifies the type of optimizer to use while training ('sgd' or 'adam')
#					epochs (list of int): Each element specifies the number of iterations over the training set
# X_train (numpy.ndarray): A matrix containing w2v representations for training set of shape (len(reviews), dim)
# y_train (numpy.ndarray): A numpy vector containing (0/1) labels corresponding to the representations in X_train of shape (X_train.shape[0], )
# X_val (numpy.ndarray): A matrix containing w2v representations for validation set of shape (len(reviews), dim)
# y_val (numpy.ndarray): A numpy vector containing (0/1) labels corresponding to the representations in X_val of shape (X_val.shape[0], )
#
# Returns:
# best_params (dict): A dictionary containing the best parameter combination:
#	    				layers (int): Number of dense layers in the neural network
#	 	     			units (int): Number of units in each dense layer
#	 					loss (string): The type of loss to optimize ('binary_crossentropy' or 'mse)
#						optimizer (string): The type of optimizer to use while training ('sgd' or 'adam')
#						epochs (int): Number of iterations over the training set
def find_best_params(params, X_train, y_train, X_val, y_val):
	best_params=dict()
	layers = params['layers']
	units = params['units']
	loss_types = params['loss']
	optimizer_types = params['optimizer']
	epochs = params['epochs']


	param_combinations = []
	for layer in layers:
		for unit in units:
			for loss_type in loss_types:
				for optimizer_type in optimizer_types:
					for epoch in epochs:
						model_dict ={}
						model_dict['layers'] = layer
						model_dict['units'] = unit
						model_dict['loss'] = loss_type
						model_dict['optimizer'] = optimizer_type
						model_dict['epochs'] = epoch
						param_combinations.append(model_dict)

	# Iterate over all combinations using one or more loops
	best_accuracy = 0.0
	for param_combination in param_combinations:
		# Reset seeds and build your model
		reset_seeds()
		model = build_nn(param_combination)
		reset_seeds()
		# Train and evaluate your model, make sure you call reset_seeds before every model.fit call
		model.fit(X_train, y_train,validation_data = (X_val, y_val), epochs=param_combination['epochs'])

		pred = model.predict(X_val).flatten()
		crt_testcases=0
		total_testcases = len(pred)
		for prob in range(0,total_testcases):
			if pred[prob]>=0.5:
				pred[prob]=1
			else:
				pred[prob]=0
			if y_val[prob]==pred[prob]:
				crt_testcases+=1
		accuracy = (crt_testcases)/(total_testcases)
		if accuracy > best_accuracy:
			best_accuracy=accuracy
			best_params=param_combination

	return best_params


# Function to convert probabilities into pos/neg labels
#
# Arguments:
# probs (numpy.ndarray): A numpy vector containing probability of being positive
#
# Returns:
# pred (numpy.ndarray): A numpy vector containing pos/neg labels such that ith value in probs is mapped to ith value in pred
# 						A value is mapped to pos label if it is >=0.5, neg otherwise
def translate_probs(probs):
	pred = np.repeat('pos', probs.shape[0])
	for index,prob in enumerate(probs):
		if prob<0.5:
			pred[index]='neg'
	return pred


def main():
	# Load dataset
	data = load_data('movie_reviews.csv')

	# Extract list of reviews from the training set
	train_data = list(filter(lambda x: x['split'] == 'train', data))
	reviews_train = [r['text'] for r in train_data]

	# Compute the word2vec representation for training set
	X_train = w2v_rep(reviews_train)
	# Save these representations in q1-train-rep.npy for submission
	np.save('q1-train-rep.npy', X_train)
l

	validation_data = list(filter(lambda x: x['split'] == 'val', data))
	validation_text = [r['text'] for r in validation_data]


	X_val = w2v_rep(validation_text)

	test_data = list(filter(lambda x: x['split'] == 'test', data))
	test_text = [r['text'] for r in test_data]

	X_test = w2v_rep(test_text)

	y_train_data = list(filter(lambda x: x['split'] == 'train', data))
	y_train_label = [r['label'] for r in y_train_data]

	for index,ele in enumerate(y_train_label):
		if ele == 'pos':
			y_train_label[index] = 1
		elif ele == 'neg':
			y_train_label[index] = 0

	y_train = np.asarray(y_train_label)

	y_val_data = list(filter(lambda x: x['split'] == 'val', data))
	y_val_label = [r['label'] for r in y_val_data]
	for index,ele in enumerate(y_val_label):
		if ele == 'pos':
			y_val_label[index] = 1
		elif ele == 'neg':
			y_val_label[index] = 0

	y_val = np.asarray(y_val_label)


	# Build a feed forward neural network model with build_nn function
	params = {
		'layers': 1,
		'units': 8,
		'loss': 'binary_crossentropy',
		'optimizer': 'adam'
	}
	# reset_seeds()
	model = build_nn(params)

	params = {
		'layers': [1, 3],
		'units': [8, 16, 32],
		'loss': ['binary_crossentropy', 'mse'],
		'optimizer': ['sgd', 'adam'],
		'epochs': [1, 5, 10]
	}
	best_params = find_best_params(params, X_train, y_train, X_val, y_val)

	print("Best parameters: {0}".format(best_params))

	# Build a model with best parameters and fit on the training set
	# reset_seeds function must be called immediately before build_nn and model.fit function

	reset_seeds()
	model = build_nn(best_params)
	reset_seeds()
	model.fit(X_train, y_train, epochs=best_params['epochs'])

	pred = model.predict(X_val).flatten()


	true_pos = 0
	true_neg = 0
	fal_pos = 0
	fal_neg = 0
	num_testcase = len(pred)
	for prob in range(num_testcase):
		if pred[prob]>=0.5:
			pred[prob]=1
		else:
			pred[prob]=0
		if pred[prob]==y_val[prob]:
			if pred[prob] == 1:
				true_pos+= 1
			else:
				true_neg+= 1
		elif y_val[prob]==0 and pred[prob]==1:
			fal_pos+=1
		elif y_val[prob]==1 and pred[prob]==0:
				fal_neg+=1

	precision = true_pos/(true_pos+fal_pos)
	recall = true_pos/(true_pos+fal_neg)
	accuracy = (true_pos)/(true_pos+fal_neg)
	f1_score = (2*precision*recall)/(precision+recall)
	#
	# print("precision:",precision)
	# print("recall:",recall)
	# print("f1",f1_score)
	# print("accuracy",accuracy)

	pred = np.zeros((10))
	# Use the model to predict labels for the test set (uncomment the line below)
	pred = model.predict(X_test)

	# Translate predicted probabilities into pos/neg labels
	pred = translate_probs(pred)
	np.save('q1-pred.npy', pred)


if __name__ == '__main__':
	main()
