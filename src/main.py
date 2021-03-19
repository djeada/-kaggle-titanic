import os
import errno
import pandas as pd
from sklearn.model_selection import train_test_split

from calculate_stats import CalculateStats
from linear_regression import LinearRegression
from logisitic_regression import LogisticRegression
from multilayer_perceptron import MultilayerPerceptron

DATASET_PATH = '../datasets/train.csv'
MODELS_DIR = '../models'

def clean_data(path):
	data_frame = pd.read_csv(path)
	
	#fill missing data for age
	data_frame['Age'].fillna(data_frame['Age'].mean(), inplace=True)

	#convert to numeric
	mapping = {'male': 0, 'female': 1}
	data_frame['Sex'] = data_frame['Sex'].replace(mapping.keys(),mapping.values())

	data_frame.drop(['Cabin', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)

	save_file_name = os.path.dirname(path) + os.sep + 'titanic_cleaned.csv'
	data_frame.to_csv(save_file_name, encoding='utf-8', index=False)

	return save_file_name


def split_data(path):

	data_frame = pd.read_csv(path)

	x = data_frame.loc[:, data_frame.columns != "Survived"]
	y = data_frame.loc[:, data_frame.columns == "Survived"]

	train_test_data = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)

	dir_path = os.path.dirname(path) + os.sep

	paths = [dir_path + file_name for file_name in ['train_features.csv', 'test_features.csv', 'train_labels.csv', 'test_labels.csv']]

	for data, path in zip(train_test_data, paths):
		data.to_csv(path, index=False)

	return paths


def train_models(models, path, features_path, labels_path):

	return [model(path, features_path, labels_path).get_path() for model in models]


def compare_results(models_paths, test_features, test_labels):
	pass


def main():

	if not os.path.isfile(DATASET_PATH):
		raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), DATASET_PATH)
	
	clean_data_path = clean_data(DATASET_PATH)

	CalculateStats(clean_data_path)
	train_features, test_feature, train_labels, test_labels  = split_data(clean_data_path)	
	
	models = [LinearRegression, LogisticRegression, MultilayerPerceptron]
	results_paths = train_models(models, MODELS_DIR, train_features, train_labels)
	compare_results(results_paths, test_feature, test_labels)


if __name__ == "__main__":
	main()

