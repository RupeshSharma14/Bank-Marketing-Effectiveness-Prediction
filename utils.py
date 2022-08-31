import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from yellowbrick.model_selection import LearningCurve
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_score

rng = np.random.RandomState(0)
stratified_kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = rng)

def sine_cosine_transform(X, feature):
  month_mapping = {"jan": 0, "feb": 1, "mar": 2, "apr": 3, "may": 4, "jun": 5, "jul": 6,
                   "aug": 7, "sep": 8, "oct": 9, "nov": 10, "dec": 11}
  X[feature] = X[feature].map(month_mapping)
  X["month_sin"] = np.sin(X["month"] * (2 * np.pi / 12))
  X["month_cos"] = np.cos(X["month"] * (2 * np.pi / 12))

  return X["month_sin"], X["month_cos"]


def target_encoding(X, feature, label):
  enc_df = pd.crosstab(X[feature], X[label], normalize = "index")["yes"]
  job_mapping = dict(zip(enc_df.index, enc_df.values))
  X[feature] = X[feature].map(job_mapping)

  return X[feature]

def ordinal_encoding(X, feature):
  education_mapping = {"unknown": 0, "primary": 1, "secondary": 2, "tertiary": 3}
  X[feature] = X[feature].map(education_mapping)

  return X[feature]

def onehot_encoding(X, features):
  x = pd.get_dummies(X, columns = features, drop_first = True)
  return x

def OneHot_encoding(X, features):
    x = pd.get_dummies(X, columns = features)
    col_names = ['age', 'job', 'education', 'balance', 'day', 'campaign', 'pdays',
             'previous', 'month_sin', 'month_cos', 'marital_married',
             'marital_single', 'default_yes', 'housing_yes', 'loan_yes',
             'poutcome_other', 'poutcome_success', 'poutcome_unknown',
             'contact_telephone', 'contact_unknown']
    for col in x.columns:
        if col not in col_names:
            x.drop(columns = col, inplace = True)
    for col in col_names:
        if col not in x.columns:
            x[col] = 0
            
    x = x[col_names]
    
    return x

def handling_outliers(row, column_name, max_thresh, min_thresh = -1):
  if row[column_name] <= min_thresh:
    return min_thresh
  elif row[column_name] <= max_thresh:
    return row[column_name]
  else:
    return max_thresh

def evaluate(model, data_prepared, label, cross_val = False):
  if cross_val:
    model_scores = cross_val_score(model, data_prepared, label, scoring = "balanced_accuracy", cv = stratified_kfold)
    score = np.mean(model_scores)
  else:
    label_pred = model.predict(data_prepared)
    score = balanced_accuracy_score(label, label_pred)
    
  return score

def plot_learning_curves(model, data_prepared, label):
  visualizer = LearningCurve(model, scoring = "balanced_accuracy", cv = stratified_kfold)
  visualizer.fit(data_prepared, label) 
  
  visualizer.show()
