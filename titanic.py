# Importing the necessary libraries for data analysis and visualization
import math
import seaborn as sns
from sklearn import tree
import numpy as np  # Import NumPy for numerical operations
import pandas as pd  # Import Pandas for data manipulation
import seaborn as sns  # Import Seaborn for data visualization
from sklearn import tree
import plotly.express as px  # Import Plotly Express for interactive data visualization
import matplotlib.pyplot as plt  # Import Matplotlib for data visualization
from scipy.stats import loguniform
from sklearn.pipeline import Pipeline  # Import Scikit-Learn's Pipeline for automating processes
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Import data scaling functions
from sklearn.model_selection import train_test_split, GridSearchCV  # Import functions for splitting data and hyperparameter tuning
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score  # Import evaluation metrics
from sklearn.linear_model import LogisticRegression  # Import Logistic Regression model
from sklearn.neighbors import KNeighborsClassifier  # Import K-Nearest Neighbors model
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree model
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest model

sns.set()  # Set the default style for Seaborn

# Importing the dataset
df = pd.read_csv("Titanic-Dataset.csv")

print("Shape of the data:- ",df.shape)

def count_plot(feature):
    # This function takes a feature as input and creates a count plot
    sns.countplot(x=feature, data=df)
    plt.show()
    print("\n")
    
columns = ['Survived','Pclass','Sex','SibSp','Embarked', 'Parch']
for i in columns:
    count_plot(i)
    
df["Age"].plot(kind='hist', title = "Age")
df.groupby('Pclass')['Survived'].value_counts()
df.groupby('Embarked')['Survived'].value_counts()

survived_counts = df['Survived'].value_counts().reset_index()
survived_counts.columns = ['Survived', 'Count']
fig = px.pie(survived_counts, values='Count', names=['No', 'Yes'], title='Survived', labels={'Count': 'Count'}, color = ['No', 'Yes'])
fig.update_traces(textposition='inside',  textinfo='percent+label+value')
fig.update_layout(uniformtext_minsize=14, uniformtext_mode='hide')
fig.show()

# Show histogram chart of survival counts by gender
fig1 = px.histogram(df, x='Sex', color='Survived', barmode='group', color_discrete_map={0: "red", 1: "blue"})
fig1.update_layout(title='Sex: Survived vs Dead')
fig1.show()

# Show histogram chart of survival counts by Pclass
fig2 = px.histogram(df, x='Pclass', color='Survived', barmode='group', title='Pclass: Survived vs Dead', labels={'Pclass': 'Pclass'}, color_discrete_map={0: 'red', 1: 'blue'})
fig2.update_layout(title='PClass: Survived vs Dead')


df.head()
df_clean = df.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)
df_clean.shape
df_clean.head()
df_clean.drop_duplicates(inplace=True)
df_clean.reset_index().shape

# Fill missing values in age column by imputing the mean
df_clean['Age'].fillna(df['Age'].mean(), inplace=True)
df_clean.isna().sum()

sex_map = {'male': 1, 'female': 2}
df_clean['Sex'] = df_clean['Sex'].map( sex_map).astype(int)

df_clean['Sex'].unique()

# Fill missing values in embarked column by imputing the mode
df_clean["Embarked"].fillna(df_clean["Embarked"].mode()[0], inplace=True)

df_clean.info()

# Transform categorical data into numerical data manually as there are only 2 to 3 values for each column
Embarked_map = {'S': 1, 'C': 2, 'Q': 3}
df_clean['Embarked'] = df_clean['Embarked'].map(Embarked_map).astype(int)

df_clean['Embarked'].unique()

## Remove Outliers 
for i in [i for i in df_clean.columns]:
    if df_clean[i].nunique()>=12:
        Q1 = df_clean[i].quantile(0.20)
        Q3 = df_clean[i].quantile(0.80)
        IQR = Q3 - Q1
        IQR = Q3 - Q1
        df_clean = df_clean[df_clean[i] <= (Q3+(1.5*IQR))]
        df_clean = df_clean[df_clean[i] >= (Q1-(1.5*IQR))]
df_clean = df_clean.reset_index(drop=True)
df_clean.info()

#Understanding the relationship between all the features
sns.pairplot(df_clean, hue='Survived')


# Let's check the correlation between the variables 
plt.figure(figsize=(20,18)) 
sns.heatmap(df_clean.corr(), annot=True, linewidths=.5) 

# Calculate the correlation list
target_corr = df_clean.corr()['Survived'].abs().sort_values(ascending=False)
# Create a bar chart to visualize the correlations
plt.figure(figsize=(10, 6))
sns.barplot(x=target_corr.index[1:], y=target_corr.values[1:])
plt.xticks(rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Correlation with diagnosis')
plt.title('Correlation between diagnosis and Features')
plt.tight_layout()
plt.show()

X = df_clean.drop("Survived", axis=1)
y = df_clean["Survived"]

print(f"'X' shape: {X.shape}")
print(f"'y' shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = True, random_state=1)

#Feature Scaling (Standardization)
std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.transform(X_test )

pd.DataFrame(X_train,columns = X.columns ).describe(include = 'all')

def model_stats(y_pred, y_test):
    
    result = np.vstack((y_pred, y_test)).T
    #print(result)
    differences = np.count_nonzero(result.sum(axis = 1) == 1 )
    print('Wrong Predictions = ',differences)
    cm = confusion_matrix(y_test, y_pred)
    print(cm, '\n Accuracy Score = ',accuracy_score(y_test, y_pred))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    
# Logistic Regression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
model_stats(y_test, y_pred)


# Create a grid of hyperparameter values
param_grid = {
    'solver': ['lbfgs', 'newton-cg', 'sag', 'saga'],
    'penalty': ['l1', 'l2'],
    'C': [1, 10, 100]
}

# Create a logistic regression classifier
classifier = LogisticRegression()

# Create a grid search object
grid_search = GridSearchCV(classifier, param_grid, cv=5)

# Fit the grid search object to the training data
grid_search.fit(X_train, y_train)

# Check if the best parameters have been set
if grid_search.best_params_ is not None:
    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # Create a logistic regression classifier with the best hyperparameters
    classifier = LogisticRegression(**best_params)
    classifier.fit(X_train, y_train)
    # Evaluate the classifier on the test data
    y_pred = classifier.predict(X_test)
    
    model_stats(y_pred,y_test)
else:
    print('The best parameters have not been set yet.')
    

# KNN

# Create a grid of hyperparameter values
param_grid = {
    "n_neighbors": [i for i in range(1,20, 1)],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'cosine']
}

# Create a KNeighborsClassifier object
knn = KNeighborsClassifier()

# Create a GridSearchCV object
grid_search = GridSearchCV(knn, param_grid,  refit=True, verbose=1, cv=5)

# Fit the grid search object to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Create a KNeighborsClassifier object with the best hyperparameters
knn = KNeighborsClassifier(**best_params)
knn.fit(X_train,y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test)

# Evaluate the predictions
model_stats(y_test,y_pred)


# KNN Minkowski Method

param_grid = {"n_neighbors": [i for i in range(1,20, 1)],
             "weights": ["uniform", "distance"],
             "algorithm": ["ball_tree", "kd_tree", "brute"],
             "leaf_size": [1, 5, 10 ],
             "p": [1,2]}
# Create a KNeighborsClassifier object
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

# Create a GridSearchCV object
grid_search = GridSearchCV(knn, param_grid,  refit=True, verbose=1, cv=5)

# Fit the grid search object to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Create a KNeighborsClassifier object with the best hyperparameters
knn = KNeighborsClassifier(**best_params)
knn.fit(X_train,y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test)

# Evaluate the predictions
model_stats(y_test,y_pred)

# Create a grid of hyperparameter values
param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [3, 5, 7, None],
    "max_features": [i for i in range(1, 10, 1)],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [i for i in range(1, 5, 1)],
    'max_leaf_nodes': [None, 10, 20],
    'min_impurity_decrease': [1e-7, 1e-5, 1e-3]
}

# Decision Tree
# Create a decision tree classifier object
classifier = DecisionTreeClassifier(random_state = 0)

# Create a GridSearchCV object
grid_search = GridSearchCV(classifier, param_grid, cv=5)

# Fit the grid search object to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Create a decision tree classifier object with the best hyperparameters
classifier = DecisionTreeClassifier(**best_params)
classifier.fit(X_train,y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test)

# Evaluate the predictions
model_stats(y_test,y_pred)

# Random Forest

# Create a grid of hyperparameter values
param_grid = {'bootstrap': [True, False],
    'n_estimators': [30,50,100],
    'max_depth': [10, 50, 100, None],
    'min_samples_split': [2, 5, 15, 30],
    'min_samples_leaf': [1, 3, 5, 10]
}

# Create a RandomForestClassifier object
classifier = RandomForestClassifier()

# Create a GridSearchCV object
grid_search = GridSearchCV(classifier, param_grid, cv=5)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Create a RandomForestClassifier object with the best hyperparameters
classifier = RandomForestClassifier(**best_params)
classifier.fit(X_train,y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test)

# Evaluate the predictions
model_stats(y_test,y_pred)