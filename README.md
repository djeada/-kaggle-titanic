# Kaggle-Titanic
Exemplary solution to Kaggle's Data Science competition: Titanic - Machine Learning from Disaster.

This is a binary classification issue in which we forecast whether or not Titanic passengers survived. 

![Capture](https://user-images.githubusercontent.com/37275728/186237499-f7436ca6-7241-41b6-b4b8-f5d3e9648311.PNG)

## Introduction

> The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

> One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.

> In this contest, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

> This Kaggle Getting Started Competition provides an ideal starting place for people who may not have a lot of experience in data science and machine learning."

<a href="https://www.kaggle.com/c/titanic">Read more.</a>

## Installation

To set up the project, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/djeada/kaggle-titanic.git
    ```

2. **Install `virtualenv`**:
    If `virtualenv` is not already installed, you can install it using:
    ```bash
    pip install virtualenv
    ```

3. **Set Up the Virtual Environment**:
    Open the terminal in the project directory and run the following commands to create and activate a virtual environment:
    ```bash
    cd kaggle-titanic
    virtualenv env
    source env/bin/activate
    ```

4. **Install Dependencies**:
    Install the required packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

5. **Navigate to Source Directory**:
    Change to the source directory:
    ```bash
    cd src
    ```

6. **Run the Main Script**:
    Execute the main Python script:
    ```bash
    python3 main.py
    ```

## Dataset Description

This dataset provides detailed information about the passengers aboard the Titanic, including demographic details, ticket information, and survival status. Here is a description of each column:

- **PassengerId**: A unique identifier for each passenger.
- **Survived**: Survival status (0 = No, 1 = Yes), indicating whether the passenger survived the disaster.
- **Pclass**: Passenger class (1 = 1st class, 2 = 2nd class, 3 = 3rd class), representing the socio-economic status.
- **Name**: Full name of the passenger.
- **Sex**: Gender of the passenger.
- **Age**: Age of the passenger.
- **SibSp**: Number of siblings and/or spouses aboard the Titanic.
- **Parch**: Number of parents and/or children aboard the Titanic.
- **Ticket**: Ticket number.
- **Fare**: Amount of money the passenger paid for the ticket.
- **Cabin**: Cabin number, if available.
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

### Sample Data

Here are the first five entries in the dataset:

| PassengerId | Survived | Pclass | Name                                              | Sex    | Age | SibSp | Parch | Ticket          | Fare    | Cabin | Embarked |
|-------------|----------|--------|---------------------------------------------------|--------|-----|-------|-------|-----------------|---------|-------|----------|
| 1           | 0        | 3      | Braund, Mr. Owen Harris                           | male   | 22  | 1     | 0     | A/5 21171       | 7.25    |       | S        |
| 2           | 1        | 1      | Cumings, Mrs. John Bradley (Florence Briggs Thayer)| female | 38  | 1     | 0     | PC 17599        | 71.2833 | C85   | C        |
| 3           | 1        | 3      | Heikkinen, Miss. Laina                            | female | 26  | 0     | 0     | STON/O2. 3101282| 7.925   |       | S        |
| 4           | 1        | 1      | Futrelle, Mrs. Jacques Heath (Lily May Peel)      | female | 35  | 1     | 0     | 113803          | 53.1    | C123  | S        |
| 5           | 0        | 3      | Allen, Mr. William Henry                          | male   | 35  | 0     | 0     | 373450          | 8.05    |       | S        |

## Potential Analyses

This dataset allows for a variety of analyses, providing insights into different aspects of the Titanic disaster and the passengers on board. Here are some of the key analyses that can be performed:

- **Survival Analysis**:
  - Investigate the factors influencing survival rates. This involves analyzing how different variables such as passenger class, gender, age, and embarkation point affected the likelihood of survival.
  - For example, we can determine if first-class passengers had a higher survival rate compared to those in third class, or if women and children were more likely to survive than men.
  - We can also explore the survival rates based on the port of embarkation to see if there were any differences in survival chances depending on where passengers boarded the ship.

- **Demographic Analysis**:
  - Examine the distribution of passengers by age, gender, and class.
  - Understand the age range of the passengers and identify which age groups were most represented.
  - Analyze the gender distribution to see the ratio of male to female passengers and how it varied across different classes.

- **Economic Analysis**:
  - Analyze the fare distribution and its correlation with survival and passenger class.
  - Determine if passengers who paid higher fares had better survival rates.
  - Explore the economic differences between passengers in different classes by looking at the fare amounts.

- **Family Analysis**:
  - Explore the impact of family size (SibSp and Parch) on survival rates.
  - Investigate whether passengers traveling with family members had a higher chance of survival compared to those traveling alone.
  - Analyze the composition of families on board and how the presence of siblings, spouses, parents, and children affected survival outcomes.

- **Text Analysis**:
  - Analyze patterns in passenger names and ticket numbers.
  - Identify common names and titles (e.g., Mr., Mrs., Miss) and see if they provide any insights into the passengers' backgrounds.
  - Study ticket numbers to see if there are any patterns related to ticket types or purchasing groups.

- **Geographic Analysis**:
  - Study the embarkation points and their relation to survival and demographics.
  - Analyze the distribution of passengers based on their port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).
  - Explore if the embarkation point had any significant impact on the survival rates or the demographics of the passengers.

### Focus on Survival Analysis

For this project, we will place a particular emphasis on **Survival Analysis**. This will involve:

1. **Data Preprocessing**: Cleaning and preparing the data to ensure accuracy and completeness for analysis.
2. **Exploratory Data Analysis (EDA)**: Visualizing the survival rates based on different factors such as class, gender, age, and embarkation point.
3. **Statistical Testing**: Conducting statistical tests to determine the significance of various factors in influencing survival rates.
4. **Modeling**: Building predictive models to identify the key determinants of survival and to predict the likelihood of survival for different passenger profiles.
5. **Interpretation**: Interpreting the results to understand the underlying reasons behind the observed survival patterns and drawing meaningful conclusions from the analysis.

### Data Preprocessing and Cleaning

1. **Setting Up Paths and Imports**:
    - Imported necessary libraries and modules for data handling, model training, and evaluation.
    - Defined paths to the dataset and test dataset files.

2. **Preprocessing Steps**:
    - Specified features to be dropped: `"Cabin"`, `"Embarked"`, `"Name"`, `"Ticket"`, `"PassengerId"`.
    - Created a directory for output files.

3. **Cleaning the Dataset**:
    - Read the dataset from the CSV file.
    - Applied a series of data cleaning filters:
        - **DropFeaturesFilter**: Removed unnecessary features.
        - **EncodeCategoricalVariablesFilter**: Encoded categorical variables to numerical values.
        - **FillMissingValuesFilter**: Filled in missing values in the dataset.

4. **Splitting the Dataset**:
    - Separated the cleaned dataset into features (`x_dataset`) and labels (`y_dataset`).
    - Split the data into training and testing sets, saving the split datasets for future use.

### Model Training

5. **Initializing Models**:
    - Defined a list of model types to be trained:
        - `LinearRegression`
        - `MultilayerPerceptron`
        - `RandomForest`
        - `SVM`
        - `LogisticRegression`
        - `GradientBoost`
    - Created an empty list to store the trained models.

6. **Training Models**:
    - Iterated through each model type, initialized, and trained each model using the training dataset (`train_x` and `train_y`).
    - Stored each trained model in the `models` list.

### Model Testing and Evaluation

7. **Evaluating Models**:
    - Initialized an empty dictionary `scores` to store evaluation metrics.
    - For each trained model, generated predictions on the testing dataset (`test_x`).
    - Calculated evaluation metrics: `accuracy`, `precision`, and `recall` for each model.
    - Stored these metrics in the `scores` dictionary and printed the results.

8. **Identifying the Best Model**:
    - Determined the best model based on the highest `accuracy` score.
    - Printed the name of the best-performing model.

### Saving the Model and Generating Predictions

9. **Saving the Best Model**:
    - Saved the best model to the output directory.

10. **Preparing Test Dataset for Predictions**:
    - Read the test dataset from the CSV file.
    - Applied the same data cleaning filters as used in the training dataset.
    - Dropped the label columns from the test dataset.
    - Generated predictions using the best model.

11. **Creating Submission File**:
    - Re-loaded the test dataset to include the `PassengerId` column.
    - Added the predicted `Survived` values to the test dataset.
    - Saved the final predictions to a CSV file named `submission.csv` in the output directory.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
