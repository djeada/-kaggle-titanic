import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

RESOURCES_PATH = '../resources/'

def autolabel(ax, rects, percentages):

	for i, rect in enumerate(rects):
		height = rect.get_height()
		ax.annotate('{}%'.format(percentages[i]),
			xy=(rect.get_x() + rect.get_width() / 2, height),
			xytext=(0, 3),
			textcoords="offset points",
			ha='center', va='bottom')


class CalculateStats():
	def __init__(self, path):
		self.calculate_statistics(path)

	def calculate_statistics(self, path):
		data_frame = pd.read_csv(path)
		self.plot_survivors_vs_deceased(data_frame)
		self.plot_survivability_by_gender(data_frame)

	def plot_survivors_vs_deceased(self, data_frame):
		data = data_frame["Survived"].values
		survivors_num = (data == 1).sum()
		deceased_num = (data == 0).sum()

		labels = ['']

		x = np.arange(len(labels))
		width = 0.5

		fig, ax = plt.subplots()
		rects1 = ax.bar(x - width/2, [survivors_num], width, label='Survivors')
		rects2 = ax.bar(x + width/2, [deceased_num], width, label='Deceased')

		ax.set_ylabel('Number of people')
		ax.set_title('Survivors vs Deceased')
		ax.set_xticks(x)
		ax.set_xticklabels(labels)
		ax.legend()


		percentage_1 = [int(survivors_num*100/(survivors_num + deceased_num))]
		percentage_2 = [int(deceased_num*100/(survivors_num + deceased_num))]

		autolabel(ax, rects1, percentage_1)
		autolabel(ax, rects2, percentage_2)

		fig.tight_layout()
		plt.savefig(RESOURCES_PATH + 'survivors_vs_deceased.png')

	def plot_survivability_by_gender(self, data_frame):

		survivors_data = data_frame["Survived"].values
		gender_data = data_frame["Sex"].values
		males_num = (gender_data == "male").sum()
		females_num = (gender_data == "female").sum()

		survived_males = 0
		survived_females = 0

		for survived, sex in zip(survivors_data, gender_data):
			if survived and sex == "male":
				survived_males += 1

			elif survived and sex == "female":
				survived_females += 1

		deceased_males = males_num - survived_males
		deceased_females = females_num - survived_females

		labels = ['men', 'women']

		x = np.arange(len(labels))
		width = 0.45

		fig, ax = plt.subplots()
		rects1 = ax.bar(x - width/2, [survived_males, survived_females], width, label='Survivors')
		rects2 = ax.bar(x + width/2, [deceased_males, deceased_females], width, label='Deceased')

		ax.set_ylabel('Number of people')
		ax.set_title('Survivability by gender')
		ax.set_xticks(x)
		ax.set_xticklabels(labels)
		ax.legend()

		percentage_1 = [int(survived_males*100/(males_num)), int(survived_females*100/(females_num))]
		percentage_2 = [int(deceased_males*100/(males_num)), int(deceased_females*100/(females_num))]

		autolabel(ax, rects1, percentage_1)
		autolabel(ax, rects2, percentage_2)

		fig.tight_layout()
		plt.savefig(RESOURCES_PATH + 'survivability_by_gender.png')


	

