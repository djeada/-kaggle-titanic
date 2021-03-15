import os
import pandas as pd
from sklearn.model_selection import train_test_split

from calculate_stats import CalculateStats

DATASET_PATH = '../datasets/train.csv'

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


def train_models(x_train, x_test, y_train, y_test):
	pass


def compare_results():
	pass


def main():
	CalculateStats(DATASET_PATH)
	clean_data_path = clean_data(DATASET_PATH)
	paths = split_data(clean_data_path)	
	train_models(*paths)
	compare_results()


if __name__ == "__main__":
	main()

