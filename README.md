# Kaggle-Titanic
Exemplary solution to Kaggle's Data Science competition: Titanic - Machine Learning from Disaster.

<h1>Introduction</h1>

> The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

> One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.

> In this contest, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

> This Kaggle Getting Started Competition provides an ideal starting place for people who may not have a lot of experience in data science and machine learning."

<a href="https://www.kaggle.com/c/titanic">Read more.</a>

<h1>Installation</h1>

Follow the steps:

- Download this repository: ```bash git clone https://github.com/djeada/kaggle-titanic.git```
- Install <i>virtualenv</i>.
- Open the terminal in the project directory and run following commands:

```bash
virtualenv env
source env/bin/activate
pip3 install -r requirements.txt
cd src
python3 main.py
```
<h1>Cleaning Data</h1>

Following steps had been taken:

- Converting non-numeric data to 1's and 0's.
- Filling missing values in <i>Age</i> column with the mean.
- Disregarding random features. <i>Cabin</i>, <i>Embarked</i>, <i>Name</i> and <i>Ticket</i> are not important in predicting preson's chances of surviving.

<h1>General Statistics</h1>
It is always a good idea to take a look at some basic statistics before using any machine learning. Some trends in the data might be obvious and could help us later in understanding predictions of different machine learning algorithms.

<h2>Survivors vs deceased</h2>

![alt text](https://github.com/djeada/kaggle-titanic/blob/main/resources/survivors_vs_deceased.png)

More peopele lost their lives than survived.

<h2>Survivability by gender</h2>

![alt text](https://github.com/djeada/kaggle-titanic/blob/main/resources/survivability_by_gender.png)

Women had slightly better chances at surviving than men.

<h2>Survivability by class</h2>

![alt text](https://github.com/djeada/kaggle-titanic/blob/main/resources/survivability_by_class.png)

Passengers from higher classes had significantly better chances at surviving.

<h2>Survivability by age</h2>

![alt text](https://github.com/djeada/kaggle-titanic/blob/main/resources/survivability_by_age.png)

For almost all age intervals the number of deceased was greater than the number of survivors. On the side note, we can see that the number of people vs age distribution has a bell shape. That observation is consistent with the central limit theorem.

<h1>Chosen models</h1>

Following models were chosen:

<h2> Linear Regression </h2>

Explanatory variables and a continuous response are modelled as linear relationships.

<h2> Logistic Regression </h2>

The probability that Y belongs to a binary class is expected (1 or 0). Fits the data to a logistic (sigmoid) function that maximizes the probability that the observations will follow the curve. In the exponent, regularization may be applied.

<h2> Multilayer Perceptron </h2>

To get to an output, it feeds inputs through various hidden layers and relies on weights and nonlinear functions.

<h2> Random Forest </h2>

Creates a tree ensemble that votes on the final forecast.

<h2> Support Vector Machine </h2>

By maximizing the margin between the hyperplane and the nearest data points of any class, data is separated into two classes.

<h2> Gradient Boosting </h2>
Gradient descent is used to train sequential models by minimizing a given loss function at each phase.

1. Begin by estimating the response's average value. 
1. Build a tree out of the errors, limited by depth or the number of leaf nodes. 
1. Use a constant learning rate to scale decision trees. 
1. Keep practicing and weighting decision trees until they reach a point of convergence.

<h1>Results</h1>

![alt text](https://github.com/djeada/kaggle-titanic/blob/main/resources/model_comparison.png)

Except for latency, <i>Random Forest</i> had won in every category. All of the chosen models, however, produced satisfactory results.
