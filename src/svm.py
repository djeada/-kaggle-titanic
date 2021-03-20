import joblib
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class SVM:
	def __init__(self, save_path, features_path, values_path):
		self.create_model(features_path, values_path)
		self.save_results(save_path)	

	def create_model(self, features_path, values_path):

		features = pd.read_csv(features_path)
		labels = pd.read_csv(values_path)

		model = SVC()
		parameters = {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10]}

		self.results = GridSearchCV(model, parameters, cv=3)
		self.results.fit(features, labels.values.ravel())

	def save_results(self, save_path):
		if not os.path.exists(save_path):
			os.makedirs(save_path)

		self.path = os.path.join(save_path, self.__class__.__name__ + '.pkl')
		joblib.dump(self.results.best_estimator_, self.path)

	def mean(self):
		return round(self.results.cv_results_['mean_test_score'], 3)

	def std(self):
		return round(self.results.cv_results_['std_test_score'], 3)

	def results(self):
		return self.results.cv_results_['params']

	def get_path(self):
		if self.path is not None:
			return self.path

		return ''

