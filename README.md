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
env/bin/activate
pip3 install -r requirements.txt
cd src
python3 main.py
```

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

<h2> Logistic Regression </h2>

<h2> Multilayer Perceptron </h2>

<h2> Random Forest </h2>

<h2> Support Vector Machine </h2>

<h2> Gradient Boosting </h2>

<h1>Results</h1>

![alt text](https://github.com/djeada/kaggle-titanic/blob/main/resources/model_comparison.png)

Overall <i>Random Forest</i> had won in all categories, except for latency. However satisfactory results were achieved for all chosen models.
