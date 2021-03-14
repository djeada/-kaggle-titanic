from calculate_stats import CalculateStats
import pandas as pd
import os

DATASET_PATH = '../datasets/train.csv'

def clean_data(path):
	data_frame = pd.read_csv(path)
	
	#fill missing data for age
	data_frame['Age'].fillna(data_frame['Age'].mean(), inplace=True)

	#convert to numeric
	#mapping = {'male': 0, 'female': 1}

	data_frame.drop(['Cabin', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)

	save_file_name = os.path.dirname(path) + os.sep + 'titanic_cleaned.csv'
	data_frame.to_csv(save_file_name, encoding='utf-8', index=False)



def train_models():
	pass


def compare_results():
	pass


def main():
	CalculateStats(DATASET_PATH)
	clean_data(DATASET_PATH)
	train_models()
	compare_results()


if __name__ == "__main__":
	main()

