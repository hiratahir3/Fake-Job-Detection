#Step#1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from wordcloud import WordCloud
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import nltk
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import spacy

spacy.cli.download("en_core_web_sm")

#step#2
# Read the CSV file and load it into a dataframe
df = pd.read_csv('fake_job_postings.csv')

# Explore the dataset
print("Columns in the dataset:")
print(df.columns)

print("\nData types:")
print(df.dtypes)

print("\nSample records:")
print(df.head())
#step#3
# Handle missing values
df = df.dropna()  # Remove rows with missing values

# Data cleaning and standardization
# Perform necessary cleaning and standardization steps on relevant columns
df['description'] = df['description'].apply(lambda x: x.lower())  # Convert description to lowercase
df['requirements'] = df['requirements'].apply(lambda x: x.lower())  # Convert requirements to lowercase
df['company_profile'] = df['company_profile'].apply(lambda x: x.lower())  # Convert company_profile to lowercase

# Convert categorical variables into numerical representations
encoder = LabelEncoder()
categorical_cols = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# Print the preprocessed dataframe
print(df.head())
#step#4
# Load the English language model in spaCy
nlp = spacy.load('en_core_web_sm')

# Initialize the PorterStemmer from NLTK
stemmer = nltk.stem.PorterStemmer()

# Apply stopword removal, lemmatization, and stemming
def preprocess_text(text):
    doc = nlp(text)
    processed_text = ' '.join([token.lemma_ for token in doc if token.text.lower() not in STOP_WORDS])
    processed_text = ' '.join([stemmer.stem(word) for word in nltk.word_tokenize(processed_text)])
    return processed_text

# Apply text processing on relevant columns
df['description'] = df['description'].apply(preprocess_text)
df['requirements'] = df['requirements'].apply(preprocess_text)
df['company_profile'] = df['company_profile'].apply(preprocess_text)

# Word frequency and word cloud
text_data = ' '.join(df['description'].values)  # Combine all the preprocessed text
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)

# Plot the word cloud
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
#step#5
# Extracting relevant features
df['description_length'] = df['description'].apply(lambda x: len(x.split()))
df['has_company_logo'] = df['has_company_logo'].astype(int)
df['has_questions'] = df['has_questions'].astype(int)

# Combining textual features with other numerical and categorical features
features = ['description_length', 'has_company_logo', 'has_questions']
textual_features = ['description', 'requirements', 'company_profile']
categorical_features = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']

# Create a comprehensive feature set
X_text = df[textual_features].values
X_numerical = df[features].values
X_categorical = pd.get_dummies(df[categorical_features], drop_first=True).values

X = np.concatenate((X_text, X_numerical, X_categorical), axis=1)
y = df['fraudulent'].values
#step#6
# Split the dataset into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Print the shape of the training, validation, and testing sets
print("Training set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_val.shape, y_val.shape)
print("Testing set shape:", X_test.shape, y_test.shape)
#step#7
# Instantiate the CountVectorizer
vectorizer = CountVectorizer()

# Convert X_train documents to lowercase if they are strings
X_train_lower = [doc.lower() if isinstance(doc, str) else str(doc).lower() for doc in X_train]

# Convert the X_train_lower list to a NumPy array
X_train_lower = np.array(X_train_lower)

# Fit and transform the training data
X_train_vectorized = vectorizer.fit_transform(X_train_lower)

# Instantiate the classification models
decision_tree = DecisionTreeClassifier()
random_forest = RandomForestClassifier()
svm = SVC()
logistic_regression = LogisticRegression()

# Train the models using the vectorized training data
decision_tree.fit(X_train_vectorized, y_train)
random_forest.fit(X_train_vectorized, y_train)
svm.fit(X_train_vectorized, y_train)
logistic_regression.fit(X_train_vectorized, y_train)
#step#8
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import ndcg_score

# Transform the testing data using the fitted vectorizer
X_test_lower = [doc.lower() if isinstance(doc, str) else str(doc).lower() for doc in X_test]
X_test_vectorized = vectorizer.transform(X_test_lower)

# Predict the labels for the testing data
y_pred_decision_tree = decision_tree.predict(X_test_vectorized)
y_pred_random_forest = random_forest.predict(X_test_vectorized)
y_pred_svm = svm.predict(X_test_vectorized)
y_pred_logistic_regression = logistic_regression.predict(X_test_vectorized)

# Evaluate decision tree
decision_tree_accuracy = accuracy_score(y_test, y_pred_decision_tree)
decision_tree_precision = precision_score(y_test, y_pred_decision_tree, average='weighted')
decision_tree_recall = recall_score(y_test, y_pred_decision_tree, average='weighted')
decision_tree_f1_score = f1_score(y_test, y_pred_decision_tree, average='weighted')
decision_tree_confusion_matrix = confusion_matrix(y_test, y_pred_decision_tree)
decision_tree_classification_report = classification_report(y_test, y_pred_decision_tree)

# Evaluate random forest
random_forest_accuracy = accuracy_score(y_test, y_pred_random_forest)
random_forest_precision = precision_score(y_test, y_pred_random_forest, average='weighted')
random_forest_recall = recall_score(y_test, y_pred_random_forest, average='weighted')
random_forest_f1_score = f1_score(y_test, y_pred_random_forest, average='weighted')
random_forest_confusion_matrix = confusion_matrix(y_test, y_pred_random_forest)
random_forest_classification_report = classification_report(y_test, y_pred_random_forest)

# Evaluate SVM
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm, average='weighted')
svm_recall = recall_score(y_test, y_pred_svm, average='weighted')
svm_f1_score = f1_score(y_test, y_pred_svm, average='weighted')
svm_confusion_matrix = confusion_matrix(y_test, y_pred_svm)
svm_classification_report = classification_report(y_test, y_pred_svm)

# Evaluate logistic regression
logistic_regression_accuracy = accuracy_score(y_test, y_pred_logistic_regression)
logistic_regression_precision = precision_score(y_test, y_pred_logistic_regression, average='weighted')
logistic_regression_recall = recall_score(y_test, y_pred_logistic_regression, average='weighted')
logistic_regression_f1_score = f1_score(y_test, y_pred_logistic_regression, average='weighted')
logistic_regression_confusion_matrix = confusion_matrix(y_test, y_pred_logistic_regression)
logistic_regression_classification_report = classification_report(y_test, y_pred_logistic_regression)

# Calculate additional evaluation metrics
decision_tree_fallout_rate = decision_tree_confusion_matrix[0, 1] / (decision_tree_confusion_matrix[0, 1] + decision_tree_confusion_matrix[0, 0])
decision_tree_average_precision = average_precision_score(y_test, y_pred_decision_tree)
decision_tree_mean_average_precision = decision_tree_average_precision / len(np.unique(y_test))
decision_tree_kappa = cohen_kappa_score(y_test, y_pred_decision_tree)

random_forest_fallout_rate = random_forest_confusion_matrix[0, 1] / (random_forest_confusion_matrix[0, 1] + random_forest_confusion_matrix[0, 0])
random_forest_average_precision = average_precision_score(y_test, y_pred_random_forest)
random_forest_mean_average_precision = random_forest_average_precision / len(np.unique(y_test))
random_forest_kappa = cohen_kappa_score(y_test, y_pred_random_forest)

svm_fallout_rate = svm_confusion_matrix[0, 1] / (svm_confusion_matrix[0, 1] + svm_confusion_matrix[0, 0])
svm_average_precision = average_precision_score(y_test, y_pred_svm)
svm_mean_average_precision = svm_average_precision / len(np.unique(y_test))
svm_kappa = cohen_kappa_score(y_test, y_pred_svm)

logistic_regression_fallout_rate = logistic_regression_confusion_matrix[0, 1] / (logistic_regression_confusion_matrix[0, 1] + logistic_regression_confusion_matrix[0, 0])
logistic_regression_average_precision = average_precision_score(y_test, y_pred_logistic_regression)
logistic_regression_mean_average_precision = logistic_regression_average_precision / len(np.unique(y_test))
logistic_regression_kappa = cohen_kappa_score(y_test, y_pred_logistic_regression)

# Calculate Cumulative Gain
decision_tree_cumulative_gain = np.cumsum(np.bincount(y_test)) / len(y_test)
random_forest_cumulative_gain = np.cumsum(np.bincount(y_test)) / len(y_test)
svm_cumulative_gain = np.cumsum(np.bincount(y_test)) / len(y_test)
logistic_regression_cumulative_gain = np.cumsum(np.bincount(y_test)) / len(y_test)

# Calculate Discounted Cumulative Gain
# indices = np.arange(1, len(y_test) + 1)
# sorted_indices = np.argsort(y_test)
# decision_tree_discounted_cumulative_gain = np.cumsum((np.bincount(y_test)[sorted_indices] / np.log2(indices[sorted_indices] + 1)) / np.sum(1 / np.log2(indices + 1)))
# random_forest_discounted_cumulative_gain = np.cumsum((np.bincount(y_test)[sorted_indices] / np.log2(indices[sorted_indices] + 1)) / np.sum(1 / np.log2(indices + 1)))
# svm_discounted_cumulative_gain = np.cumsum((np.bincount(y_test)[sorted_indices] / np.log2(indices[sorted_indices] + 1)) / np.sum(1 / np.log2(indices + 1)))
# logistic_regression_discounted_cumulative_gain = np.cumsum((np.bincount(y_test)[sorted_indices] / np.log2(indices[sorted_indices] + 1)) / np.sum(1 / np.log2(indices + 1)))

# Calculate Normalized Discounted Cumulative Gain
decision_tree_ndcg = ndcg_score([y_test], [y_pred_decision_tree])
random_forest_ndcg = ndcg_score([y_test], [y_pred_random_forest])
svm_ndcg = ndcg_score([y_test], [y_pred_svm])
logistic_regression_ndcg = ndcg_score([y_test], [y_pred_logistic_regression])

# Print the evaluation metrics
print("Decision Tree:")
print("Accuracy:", decision_tree_accuracy)
print("Precision:", decision_tree_precision)
print("Recall:", decision_tree_recall)
print("F1 Score:", decision_tree_f1_score)
print("Fallout Rate:", decision_tree_fallout_rate)
print("Average Precision:", decision_tree_average_precision)
print("Mean Average Precision:", decision_tree_mean_average_precision)
print("Kappa:", decision_tree_kappa)
print("Cumulative Gain:", decision_tree_cumulative_gain)
#print("Discounted Cumulative Gain:", decision_tree_discounted_cumulative_gain)
print("Normalized Discounted Cumulative Gain:", decision_tree_ndcg)
print("Confusion Matrix:")
print(decision_tree_confusion_matrix)
print("Classification Report:")
print(decision_tree_classification_report)
print()

print("Random Forest:")
print("Accuracy:", random_forest_accuracy)
print("Precision:", random_forest_precision)
print("Recall:", random_forest_recall)
print("F1 Score:", random_forest_f1_score)
print("Fallout Rate:", random_forest_fallout_rate)
print("Average Precision:", random_forest_average_precision)
print("Mean Average Precision:", random_forest_mean_average_precision)
print("Kappa:", random_forest_kappa)
print("Cumulative Gain:", random_forest_cumulative_gain)
#print("Discounted Cumulative Gain:", random_forest_discounted_cumulative_gain)
print("Normalized Discounted Cumulative Gain:", random_forest_ndcg)
print("Confusion Matrix:")
print(random_forest_confusion_matrix)
print("Classification Report:")
print(random_forest_classification_report)
print()

print("SVM:")
print("Accuracy:", svm_accuracy)
print("Precision:", svm_precision)
print("Recall:", svm_recall)
print("F1 Score:", svm_f1_score)
print("Fallout Rate:", svm_fallout_rate)
print("Average Precision:", svm_average_precision)
print("Mean Average Precision:", svm_mean_average_precision)
print("Kappa:", svm_kappa)
print("Cumulative Gain:", svm_cumulative_gain)
#print("Discounted Cumulative Gain:", svm_discounted_cumulative_gain)
print("Normalized Discounted Cumulative Gain:", svm_ndcg)
print("Confusion Matrix:")
print(svm_confusion_matrix)
print("Classification Report:")
print(svm_classification_report)
print()

print("Logistic Regression:")
print("Accuracy:", logistic_regression_accuracy)
print("Precision:", logistic_regression_precision)
print("Recall:", logistic_regression_recall)
print("F1 Score:", logistic_regression_f1_score)
print("Fallout Rate:", logistic_regression_fallout_rate)
print("Average Precision:", logistic_regression_average_precision)
print("Mean Average Precision:", logistic_regression_mean_average_precision)
print("Kappa:", logistic_regression_kappa)
print("Cumulative Gain:", logistic_regression_cumulative_gain)
#print("Discounted Cumulative Gain:", logistic_regression_discounted_cumulative_gain)
print("Normalized Discounted Cumulative Gain:", logistic_regression_ndcg)
print("Confusion Matrix:")
print(logistic_regression_confusion_matrix)
print("Classification Report:")
print(logistic_regression_classification_report)
from sklearn.model_selection import GridSearchCV

# Define the hyperparameter grid for each selected model
param_grid_decision_tree = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

param_grid_random_forest = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

param_grid_svm = {
    'C': [1.0, 10.0, 100.0],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

param_grid_logistic_regression = {
    'C': [1.0, 10.0, 100.0],
    'solver': ['liblinear', 'saga'],
    'max_iter': [3000]
}

# Perform grid search for each model to find the best hyperparameters
grid_search_decision_tree = GridSearchCV(decision_tree, param_grid=param_grid_decision_tree, cv=5)
grid_search_decision_tree.fit(X_train_vectorized, y_train)

grid_search_random_forest = GridSearchCV(random_forest, param_grid=param_grid_random_forest, cv=5)
grid_search_random_forest.fit(X_train_vectorized, y_train)

grid_search_svm = GridSearchCV(svm, param_grid=param_grid_svm, cv=5)
grid_search_svm.fit(X_train_vectorized, y_train)

grid_search_logistic_regression = GridSearchCV(logistic_regression, param_grid=param_grid_logistic_regression, cv=5)
grid_search_logistic_regression.fit(X_train_vectorized, y_train)

# Get the best hyperparameters and models for each model
best_hyperparameters_decision_tree = grid_search_decision_tree.best_params_
best_model_decision_tree = grid_search_decision_tree.best_estimator_

best_hyperparameters_random_forest = grid_search_random_forest.best_params_
best_model_random_forest = grid_search_random_forest.best_estimator_

best_hyperparameters_svm = grid_search_svm.best_params_
best_model_svm = grid_search_svm.best_estimator_

best_hyperparameters_logistic_regression = grid_search_logistic_regression.best_params_
best_model_logistic_regression = grid_search_logistic_regression.best_estimator_

# Print the best hyperparameters and models for each model
print("Decision Tree")
print("Best Hyperparameters:", best_hyperparameters_decision_tree)
print("Best Model:", best_model_decision_tree)
print()

print("Random Forest")
print("Best Hyperparameters:", best_hyperparameters_random_forest)
print("Best Model:", best_model_random_forest)
print()

print("SVM")
print("Best Hyperparameters:", best_hyperparameters_svm)
print("Best Model:", best_model_svm)
print()

print("Logistic Regression")
print("Best Hyperparameters:", best_hyperparameters_logistic_regression)
print("Best Model:", best_model_logistic_regression)
print()

# Train the best models using the vectorized training data
best_model_decision_tree.fit(X_train_vectorized, y_train)
best_model_random_forest.fit(X_train_vectorized, y_train)
best_model_svm.fit(X_train_vectorized, y_train)
best_model_logistic_regression.fit(X_train_vectorized, y_train)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Assuming X_val and y_val contain your validation data and labels, respectively

X_val_lower = [doc.lower() if isinstance(doc, str) else str(doc).lower() for doc in X_val]
X_val_vectorized = vectorizer.transform(X_val_lower)

y_pred_decision_tree = best_model_decision_tree.predict(X_val_vectorized)
y_pred_random_forest = best_model_random_forest.predict(X_val_vectorized)
y_pred_svm = best_model_svm.predict(X_val_vectorized)
y_pred_logistic_regression = best_model_logistic_regression.predict(X_val_vectorized)

# Print classification reports for each model
print("Decision Tree Classification Report:")
print(classification_report(y_val, y_pred_decision_tree))
print()

print("Random Forest Classification Report:")
print(classification_report(y_val, y_pred_random_forest))
print()

print("SVM Classification Report:")
print(classification_report(y_val, y_pred_svm))
print()

print("Logistic Regression Classification Report:")
print(classification_report(y_val, y_pred_logistic_regression))
print()

# Plot confusion matrices for each model
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Confusion Matrices', fontsize=16)

sns.heatmap(confusion_matrix(y_val, y_pred_decision_tree), annot=True, fmt='d', cmap='Purples', ax=axes[0, 0])
axes[0, 0].set_title('Decision Tree')

sns.heatmap(confusion_matrix(y_val, y_pred_random_forest), annot=True, fmt='d', cmap='Purples', ax=axes[0, 1])
axes[0, 1].set_title('Random Forest')

sns.heatmap(confusion_matrix(y_val, y_pred_svm), annot=True, fmt='d', cmap='Purples', ax=axes[1, 0])
axes[1, 0].set_title('SVM')

sns.heatmap(confusion_matrix(y_val, y_pred_logistic_regression), annot=True, fmt='d', cmap='Purples', ax=axes[1, 1])
axes[1, 1].set_title('Logistic Regression')

plt.tight_layout()
plt.show()
