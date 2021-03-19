import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

RESOURCES_PATH = '../resources/'

def autolabel(ax, rects, percentages, font_size=14):

	for i, rect in enumerate(rects):
		height = rect.get_height()
		ax.annotate('{}%'.format(percentages[i]),
			xy=(rect.get_x() + rect.get_width() / 2, height),
			xytext=(0, 3),
			textcoords='offset points',
			ha='center', va='bottom', fontsize=font_size)


class CalculateStats():
	def __init__(self, path):
		self.calculate_statistics(path)

	def calculate_statistics(self, path):
		data_frame = pd.read_csv(path)
		self.plot_survivors_vs_deceased(data_frame)
		self.plot_survivability_by_gender(data_frame)
		self.plot_survivability_by_class(data_frame)
		self.plot_survivability_by_age(data_frame)

	def plot_survivors_vs_deceased(self, data_frame):
		data = data_frame['Survived'].values
		survivors_num = (data == 1).sum()
		deceased_num = (data == 0).sum()

		labels = ['Survivors vs Deceased']

		x = np.arange(len(labels))
		width = 0.5

		fig, ax = plt.subplots()
		rects1 = ax.bar(x - width/2, [survivors_num], width, label='Survivors')
		rects2 = ax.bar(x + width/2, [deceased_num], width, label='Deceased')

		ax.set_ylabel('Number of people')
		ax.set_title('Survivability - general')
		ax.set_xticks(x)
		ax.set_xticklabels(labels)
		ax.set_ylim([None, 1.1*max(survivors_num, deceased_num)])		
		ax.legend()


		percentage_1 = [int(survivors_num*100/(survivors_num + deceased_num))]
		percentage_2 = [int(deceased_num*100/(survivors_num + deceased_num))]

		autolabel(ax, rects1, percentage_1)
		autolabel(ax, rects2, percentage_2)

		fig.tight_layout()
		plt.savefig(RESOURCES_PATH + 'survivors_vs_deceased.png')

	def plot_survivability_by_gender(self, data_frame):

		survivors_data = data_frame['Survived'].values
		gender_data = data_frame['Sex'].values
		males_num = (gender_data == 1).sum()
		females_num = (gender_data == 0).sum()

		survived_males = 0
		survived_females = 0

		for survived, sex in zip(survivors_data, gender_data):
			if survived and sex == 0:
				survived_males += 1

			elif survived and sex == 1:
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
		ax.set_ylim([None, 1.1*max(survived_males, survived_females, deceased_males, deceased_females)])		
		ax.legend()

		percentage_1 = [int(survived_males*100/(males_num)), int(survived_females*100/(females_num))]
		percentage_2 = [int(deceased_males*100/(males_num)), int(deceased_females*100/(females_num))]

		autolabel(ax, rects1, percentage_1)
		autolabel(ax, rects2, percentage_2)

		fig.tight_layout()
		plt.savefig(RESOURCES_PATH + 'survivability_by_gender.png')


	def plot_survivability_by_class(self, data_frame):

		survivors_data = data_frame['Survived'].values
		class_data = data_frame['Pclass'].values
		first_class_total = (class_data == 1).sum()
		second_class_total = (class_data == 2).sum()
		third_class_total = (class_data == 3).sum()

		survived_first_class = 0
		survived_second_class = 0
		survived_third_class = 0

		for survived, pclass in zip(survivors_data, class_data):
			if survived and pclass == 1:
				survived_first_class += 1

			elif survived and pclass == 2:
				survived_second_class += 1

			elif survived and pclass == 3:
				survived_third_class += 1

		deceased_first_class = first_class_total - survived_first_class
		deceased_second_class = second_class_total - survived_second_class
		deceased_third_class = third_class_total - survived_third_class


		survived_by_class = [survived_first_class, survived_second_class, survived_third_class]
		deceased_by_class = [deceased_first_class, deceased_second_class, deceased_third_class]
		labels = ['First class', 'Second class', 'Third class']

		x = np.arange(len(labels))
		width = 0.45

		fig, ax = plt.subplots()
		rects1 = ax.bar(x - width/2, survived_by_class, width, label='Survivors')
		rects2 = ax.bar(x + width/2, deceased_by_class, width, label='Deceased')

		ax.set_ylabel('Number of people')
		ax.set_title('Survivability by class')
		ax.set_xticks(x)
		ax.set_xticklabels(labels)
		ax.set_ylim([None, 1.1*max(max(survived_by_class), max(deceased_by_class))])		
		ax.legend()

		percentage_1 = [int(survived_first_class*100/(first_class_total)), int(survived_second_class*100/(second_class_total)), int(survived_third_class*100/(third_class_total))]
		percentage_2 = [int(deceased_first_class*100/(first_class_total)), int(deceased_second_class*100/(second_class_total)), int(deceased_third_class*100/(third_class_total))]

		autolabel(ax, rects1, percentage_1)
		autolabel(ax, rects2, percentage_2)

		fig.tight_layout()
		plt.savefig(RESOURCES_PATH + 'survivability_by_class.png')


	def plot_survivability_by_age(self, data_frame):

		survivors_data = data_frame['Survived'].values
		data_frame['Age'] = data_frame['Age'].astype(int)
		age_data = data_frame['Age'].values

		max_age = max(age_data)

		n = 5
		upper_limit = max_age + max_age % n + 1

		total_for_age_interval = {i : ( age_data <= i).sum() for i in range(n, upper_limit, n)}

		for age_interval in range(upper_limit-1, n, -n):
			total_for_age_interval[age_interval] -= total_for_age_interval[age_interval-n]

		age_to_age_interval = {i + n*j + 1 :  (j+1)*n  for j in range(upper_limit//n)  for i in range(n) }		
		age_to_age_interval[0] = n
		
		survivors_for_age_interval = {age_interval : 0 for age_interval in total_for_age_interval}

		for survived, age in zip(survivors_data, age_data):

			age_interval = age_to_age_interval[age]

			if survived:
				survivors_for_age_interval[age_interval] += 1

		deceased_for_age_interval = {age_interval : total - survivors_for_age_interval[age_interval] for age_interval, total in total_for_age_interval.items()}

		labels = [str(i - n + 1) + '-' + str(i) for i in range(n, upper_limit, n)]

		x = np.arange(len(labels))
		width = 0.4

		fig, ax = plt.subplots(figsize=(14, 8))


		rects1 = ax.bar(x - width/2, survivors_for_age_interval.values(), width, label='Survivors')
		rects2 = ax.bar(x + width/2, deceased_for_age_interval.values(), width, label='Deceased')
		
		ax.set_title('Survivability by age', fontsize=16)
		ax.set_ylabel('Number of people', fontsize=14)
		ax.set_xlabel('Age interval', fontsize=14)
		ax.set_xticks(x)
		ax.set_xticklabels(labels)
		ax.set_ylim([None, 1.1*max(deceased_for_age_interval.values())])		
		ax.legend(prop={'size': 20})

		percentages_1 = [int(survivors_for_age_interval[age_interval]*100/total) for age_interval, total in total_for_age_interval.items()]
		percentages_2 = [int(deceased_for_age_interval[age_interval]*100/total) for age_interval, total in total_for_age_interval.items()]
		
		autolabel(ax, rects1, percentages_1, 10)
		autolabel(ax, rects2, percentages_2, 10)

		fig.tight_layout()
		plt.savefig(RESOURCES_PATH + 'survivability_by_age.png')
	

	

