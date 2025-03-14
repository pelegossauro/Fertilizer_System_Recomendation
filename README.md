# Fertilizer_System_Recomendation
Here is a README file for your project:

---

# Fertilizer Classification using Machine Learning

## Overview
This project aims to classify different types of fertilizers based on the content of three primary nutrients: Nitrogen, Potassium, and Phosphorous. The classification model is developed using the RandomForestClassifier from scikit-learn. The dataset contains various fertilizers and their respective nutrient composition. The goal is to predict the type of fertilizer based on the input values of these nutrients.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Data](#data)
4. [Model](#model)
5. [Results](#results)
6. [License](#license)

## Installation

Ensure you have Python 3.x installed. Then, install the required libraries by running the following command:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should contain the following dependencies:

```
numpy
pandas
seaborn
matplotlib
scikit-learn
```

## Usage

### Step 1: Prepare the Data
Load the data from the `Fertilizer.csv` file:

```python
import pandas as pd
df = pd.read_csv("Fertilizer.csv")
```

### Step 2: Data Preprocessing
We clean and prepare the data by removing irrelevant columns, performing label encoding on the fertilizer names, and splitting the data into training and testing sets.

```python
from sklearn.model_selection import train_test_split
X = df.drop(columns=['Fertilizer Name'])
y = df['Fertilizer Name']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=42)
```

### Step 3: Feature Scaling
Standardize the feature data to ensure better performance of the model:

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

### Step 4: Train the Model
We use a Random Forest Classifier to train the model.

```python
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=42)
classifier.fit(X_train, y_train)
```

### Step 5: Evaluate the Model
Evaluate the model's performance using accuracy scores and confusion matrices.

```python
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(cm)
print(accuracy)
```

### Step 6: Hyperparameter Tuning
Use GridSearchCV to find the best parameters for the Random Forest Classifier.

```python
from sklearn.model_selection import GridSearchCV
params = {'n_estimators': [300, 400, 500], 'max_depth': [5, 6, 7], 'min_samples_split': [2, 5, 8]}
grid_rand = GridSearchCV(classifier, params, cv=3, verbose=3, n_jobs=-1)
grid_rand.fit(X_train, y_train)
```

### Step 7: Save and Load the Model
The trained model is saved using `pickle`, which can then be used for making predictions.

```python
import pickle
pickle_out = open('classifier1.pkl', 'wb')
pickle.dump(grid_rand, pickle_out)
pickle_out.close()

model = pickle.load(open('classifier1.pkl', 'rb'))
```

### Step 8: Making Predictions
Make predictions using the trained model.

```python
ans = model.predict([[12, 10, 13]])
```

## Data

The dataset (`Fertilizer.csv`) contains the following columns:
- **Nitrogen**: The nitrogen content of the fertilizer.
- **Potassium**: The potassium content of the fertilizer.
- **Phosphorous**: The phosphorous content of the fertilizer.
- **Fertilizer Name**: The type of fertilizer.

Example:

| Nitrogen | Potassium | Phosphorous | Fertilizer Name |
|----------|-----------|-------------|-----------------|
| 37       | 0         | 0           | Urea            |
| 12       | 0         | 36          | DAP             |

## Model

The model used in this project is a **Random Forest Classifier**. It is trained on the dataset after performing feature scaling and label encoding. The model predicts the fertilizer name based on the nutrient content.

The hyperparameters of the model are tuned using **GridSearchCV** to find the optimal values for:
- `n_estimators`: The number of trees in the forest.
- `max_depth`: The maximum depth of each tree.
- `min_samples_split`: The minimum number of samples required to split an internal node.

## Results

The model achieves an accuracy score of **1.0** on the test dataset after tuning. The classification report demonstrates excellent precision, recall, and f1-score for all the fertilizer types.

```bash
accuracy = 1.0
```

### Best Parameters:
- `n_estimators = 300`
- `max_depth = 5`
- `min_samples_split = 2`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This README file provides a comprehensive guide for setting up and running your project, as well as explaining the data and model used. You can adjust the content to suit your specific repository style or further elaborate on the project details.
