# Kaggle-Titanic
Exemplary solution to Kaggle's Data Science competition: Titanic - Machine Learning from Disaster.

![Capture](https://user-images.githubusercontent.com/37275728/186236919-319e8b87-8087-4b61-83ff-bfad0f0c3eb7.PNG)

## Introduction

> The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

> One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.

> In this contest, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

> This Kaggle Getting Started Competition provides an ideal starting place for people who may not have a lot of experience in data science and machine learning."

<a href="https://www.kaggle.com/c/titanic">Read more.</a>

## Installation

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
## Cleaning Data

Following steps had been taken:

- Converting non-numeric data to 1's and 0's.
- Filling missing values in <i>Age</i> column with the mean.
- Disregarding random features. <i>Cabin</i>, <i>Embarked</i>, <i>Name</i> and <i>Ticket</i> are not important in predicting preson's chances of surviving.

## General Statistics
It is always a good idea to take a look at some basic statistics before using any machine learning. Some trends in the data might be obvious and could help us later in understanding predictions of different machine learning algorithms.

### Survivors vs deceased

![alt text](https://github.com/djeada/kaggle-titanic/blob/main/resources/survivors_vs_deceased.png)

More peopele lost their lives than survived.

### Survivability by gender

![alt text](https://github.com/djeada/kaggle-titanic/blob/main/resources/survivability_by_gender.png)

Women had slightly better chances at surviving than men.

### Survivability by class

![alt text](https://github.com/djeada/kaggle-titanic/blob/main/resources/survivability_by_class.png)

Passengers from higher classes had significantly better chances at surviving.

### Survivability by age

![alt text](https://github.com/djeada/kaggle-titanic/blob/main/resources/survivability_by_age.png)

For almost all age intervals the number of deceased was greater than the number of survivors. On the side note, we can see that the number of people vs age distribution has a bell shape. That observation is consistent with the central limit theorem.

## Chosen models

Following models were chosen:

###  Linear Regression 

Explanatory variables and a continuous response are modelled as linear relationships.

###  Logistic Regression 

Y belongs to a binary class, i.e. it can only be 1 or 0. The algorithm fits the data to a logistic (sigmoid) function that maximizes the probability that the observations will follow the curve. In the exponent, regularization may be applied.

###  Multilayer Perceptron 

To get to an output, it feeds inputs through various hidden layers and relies on weights and nonlinear functions.

###  Random Forest 

Creates multiple decision trees and merges them to produce a more accurate and stable prediction.

###  Support Vector Machine 

Data is separated into two classes by maximizing the margin between the hyperplane and the nearest data points of any class.

###  Gradient Boosting 
Gradient descent is used to train sequential models by minimizing a given loss function at each phase.

1. Begin by estimating the response's average value. 
1. Build a tree out of the errors, limited by depth or the number of leaf nodes. 
1. Use a constant learning rate to scale decision trees. 
1. Keep practicing and weighting decision trees until they reach a point of convergence.

## Results

![alt text](https://github.com/djeada/kaggle-titanic/blob/main/resources/model_comparison.png)

Except for latency, <i>Random Forest</i> had won in every category. All of the chosen models, however, produced satisfactory results.
