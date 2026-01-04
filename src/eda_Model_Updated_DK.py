#!/usr/bin/env python
# coding: utf-8

# # EDA – Heart Disease UCI
# ## Load data and inspect distributions.

# In[2]:


from ucimlrepo import fetch_ucirepo

# fetch dataset
heart_disease = fetch_ucirepo(id=45)

# data (as pandas dataframes)
X = heart_disease.data.features
y = heart_disease.data.targets

# metadata
print(heart_disease.metadata)

# variable information
print(heart_disease.variables)


# In[3]:


print(y)


# In[4]:


print(X)


# # Step 1: Data Acquisition & Initial Cleaning

# ## Fetch the dataset:

# In[5]:


from ucimlrepo import fetch_ucirepo
import pandas as pd

heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets

# Combine into one DataFrame
df = pd.concat([X, y], axis=1)
df.columns = [c.strip().lower() for c in df.columns]  # normalize column names


# ## Inspect the data:

# In[6]:


print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())  # check missing values


# ## Handle missing values:
# 
# If any column has missing values, decide:
# 
# Drop rows with missing values (if few).
# Or impute (mean/median for numeric, mode for categorical).

# In[7]:


df.fillna(df.median(), inplace=True)


# ## Encode categorical features:
# 
# Identify categorical columns (e.g., sex, cp, thal).
# Use OneHotEncoder or pd.get_dummies() for quick EDA

# In[8]:


df_encoded = pd.get_dummies(df, drop_first=True)


# # Step 2: Basic EDA Visualizations
# ## Class balance:

# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='num', data=df)
plt.title('Class Balance')
plt.show()


# In[11]:


df.hist(figsize=(12,10))
plt.tight_layout()
plt.show()


# ## Feature Correlation

# In[12]:


corr = df.corr(numeric_only=True)
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()


# In[13]:


import os
save_dir = os.path.join(os.getcwd(), "mlOps")
os.makedirs(save_dir, exist_ok=True)


# In[14]:


from ucimlrepo import fetch_ucirepo
import pandas as pd

# Fetch UCI Heart Disease
ds = fetch_ucirepo(id=45)
X = ds.data.features.copy()
y = ds.data.targets.copy()

# Ensure y is a Series named 'num'
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]
y.name = 'num'

# Combine and normalize column names
df = pd.concat([X, y], axis=1)
df.columns = [c.strip().lower() for c in df.columns]

# Optional: make binary target (presence of heart disease)
df['target'] = (df['num'] > 0).astype(int)

# Save files
df.to_csv(os.path.join(save_dir, 'heart.csv'), index=False)               # features + num + target
df.drop(columns=['num']).to_csv(os.path.join(save_dir, 'heart_binary.csv'), index=False)  # binary target only

print('Saved:')
print(' -', os.path.join(save_dir, 'heart.csv'))
print(' -', os.path.join(save_dir, 'heart_binary.csv'))


# # Step 3 — Feature Engineering & Model Development

# In[15]:


# Core
import numpy as np
import pandas as pd

# Modeling
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Metrics & plots
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# ## 3.1 Prepare data & target

# In[16]:


# If df is NOT already defined in your notebook, uncomment the next block to load:
# from ucimlrepo import fetch_ucirepo
# heart_disease = fetch_ucirepo(id=45)
# X = heart_disease.data.features
# y = heart_disease.data.targets
# df = pd.concat([X, y], axis=1)
# df.columns = [c.strip().lower() for c in df.columns]

# Binary target: presence of heart disease (typical formulation)
# UCI 'num' is 0..4; treat >0 as disease present (1), else absent (0)
df['target'] = (df['num'] > 0).astype(int)

# Feature set (drop original multiclass target)
X = df.drop(columns=['num', 'target'])
y = df['target']


# ## 3.2 Feature typing

# In[17]:


categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
numeric_cols     = [c for c in X.columns if c not in categorical_cols]  # includes age, trestbps, chol, thalach, oldpeak, ca

print("Categorical:", categorical_cols)
print("Numeric    :", numeric_cols)


# ## 3.3 Preprocessing pipeline
# 
# Numeric: median imputation ➜ standard scaling
# Categorical: most‑frequent imputation ➜ one‑hot (ignore unseen categories in CV)

# In[18]:


numeric_tf = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

categorical_tf = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot',  OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer(
    transformers=[
        ('num',  numeric_tf,     numeric_cols),
        ('cat',  categorical_tf, categorical_cols)
    ],
    remainder='drop'  # just to be explicit
)


# ## 3.4 Model candidates & tuning grids
# We’ll optimize primarily for ROC‑AUC (good for imbalanced/binary problems) and report accuracy/precision/recall alongside.
# 
# Logistic Regression: regularization strength C, penalty l2, solver liblinear/saga, with class_weight='balanced'.
# Random Forest: n_estimators, max_depth, min_samples_leaf, max_features, with class_weight='balanced'.

# In[19]:


# # --- Pipelines
# logreg_pipe = Pipeline(steps=[
#     ('prep', preprocess),
#     ('clf',  LogisticRegression(max_iter=5000, class_weight='balanced'))
# ])

# rf_pipe = Pipeline(steps=[
#     ('prep', preprocess),
#     ('clf',  RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1))
# ])

# # --- Grids (fix typos: 'l2' not '12'; param names with __) ---
# logreg_grid = {
#     'clf__penalty': ['l2'],
#     'clf__solver':  ['liblinear', 'saga'],
#     'clf__C':       [0.01, 0.1, 1.0, 3.0, 10.0]
# }

# rf_grid = {
#     'clf__n_estimators':    [200, 400, 800],
#     'clf__max_depth':       [None, 5, 10, 20],
#     'clf__min_samples_leaf':[1, 2, 4],
#     'clf__max_features':    ['sqrt', 'log2']
# }

# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # --- GridSearch (use X, y consistently; fix n_jobs and cv) ---
# logreg_gs = GridSearchCV(
#     estimator=logreg_pipe,
#     param_grid=logreg_grid,
#     scoring='roc_auc',
#     cv=skf,
#     n_jobs=-1,
#     refit=True
# ).fit(X, y)

# rf_gs = GridSearchCV(
#     estimator=rf_pipe,
#     param_grid=rf_grid,
#     scoring='roc_auc',
#     cv=skf,
#     n_jobs=-1,
#     refit=True
# ).fit(X, y)

# print("LogReg best ROC-AUC:", logreg_gs.best_score_, "best params:", logreg_gs.best_params_)
# print("RF     best ROC-AUC:", rf_gs.best_score_,     "best params:", rf_gs.best_params_)

# # --- Final evaluation with cross_validate (consistent X, y) ---
# scoring = {'accuracy':'accuracy','precision':'precision','recall':'recall','roc_auc':'roc_auc'}

# logreg_cv = cross_validate(logreg_gs.best_estimator_, X, y, cv=skf, scoring=scoring, n_jobs=-1)
# rf_cv     = cross_validate(rf_gs.best_estimator_,     X, y, cv=skf, scoring=scoring, n_jobs=-1)

# def summarize_cv(name, cvres):
#     print(f"\n{name} (5-fold CV)")
#     for k in ['test_accuracy','test_precision','test_recall','test_roc_auc']:
#         print(f"  {k.replace('test_','').upper():10s}: mean={cvres[k].mean():.3f}  std={cvres[k].std():.3f}")

# summarize_cv("Logistic Regression", logreg_cv)
# summarize_cv("Random Forest",      rf_cv)

# # --- Out-of-fold ROC curves (fix variable names and labels) ---
# y_proba_lr = cross_val_predict(logreg_gs.best_estimator_, X, y, cv=skf, method='predict_proba', n_jobs=-1)[:, 1]
# y_proba_rf = cross_val_predict(rf_gs.best_estimator_,     X, y, cv=skf, method='predict_proba', n_jobs=-1)[:, 1]

# from sklearn.metrics import roc_curve, roc_auc_score
# fpr_lr, tpr_lr, _ = roc_curve(y, y_proba_lr)
# fpr_rf, tpr_rf, _ = roc_curve(y, y_proba_rf)
# auc_lr = roc_auc_score(y, y_proba_lr)
# auc_rf = roc_auc_score(y, y_proba_rf)

# plt.figure(figsize=(7,5))
# plt.plot(fpr_lr, tpr_lr, label=f'LogReg (AUC={auc_lr:.3f})', lw=2)
# plt.plot(fpr_rf, tpr_rf, label=f'RandomForest (AUC={auc_rf:.3f})', lw=2)
# plt.plot([0,1],[0,1],'k--',alpha=0.5)
# plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
# plt.title('Cross-validated ROC Curves'); plt.legend(); plt.grid(alpha=0.2)
# plt.show()


# In[20]:


# Base pipelines
logreg_pipe = Pipeline(steps=[
    ('prep', preprocess),
    ('clf',  LogisticRegression(max_iter=5000, class_weight='balanced', n_jobs=None))
])

rf_pipe = Pipeline(steps=[
    ('prep', preprocess),
    ('clf',  RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1))
])

# Hyperparameter grids
logreg_grid = {
    'clf__penalty': ['l2'],
    'clf__solver':  ['liblinear', 'saga'],
    'clf__C':       [0.01, 0.1, 1.0, 3.0, 10.0]
}

rf_grid = {
    'clf__n_estimators':   [200, 400, 800],
    'clf__max_depth':      [None, 5, 10, 20],
    'clf__min_samples_leaf':[1, 2, 4],
    'clf__max_features':   ['sqrt', 'log2']
}


# ## 3.5 Cross‑validation & model selection
# Use StratifiedKFold to preserve class balance across folds. Optimize by ROC‑AUC; keep refit on the best params.

# In[21]:


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Logistic Regression selection
logreg_gs = GridSearchCV(
    estimator=logreg_pipe,
    param_grid=logreg_grid,
    scoring='roc_auc',
    cv=skf,
    n_jobs=-1,
    refit=True
)
# logreg_gs.fit(X, y)

# Random Forest selection
rf_gs = GridSearchCV(
    estimator=rf_pipe,
    param_grid=rf_grid,
    scoring='roc_auc',
    cv=skf,
    n_jobs=-1,
    refit=True
)
# rf_gs.fit(X, y)

# print("LogReg best ROC-AUC:", logreg_gs.best_score_, "best params:", logreg_gs.best_params_)
# print("RF     best ROC-AUC:", rf_gs.best_score_,     "best params:", rf_gs.best_params_)


# In[22]:


import mlflow
import mlflow.sklearn
import joblib
from datetime import datetime

now = datetime.now()
datetime_string = now.isoformat()


scoring = {
    "accuracy": "accuracy",
    "precision": "precision_weighted",
    "recall": "recall_weighted",
    "f1": "f1_weighted",
    "roc_auc": "roc_auc_ovr"
}

mlflow.set_tracking_uri("http://host.docker.internal:5000")
mlflow.set_experiment("LogisticRegression_GridSearch_" + datetime_string)

# Start MLflow run
with mlflow.start_run(run_name="LogisticRegression_GridSearch_" + datetime_string):
    logreg_gs.fit(X, y)

    # Cross-validated evaluation of best model
    logreg_cv = cross_validate(
        logreg_gs.best_estimator_,
        X, y,
        cv=skf,
        scoring=scoring,
        n_jobs=-1
    )

    # Aggregate metrics
    logreg_cv_metrics = {
        "cv_accuracy": logreg_cv["test_accuracy"].mean(),
        "cv_precision": logreg_cv["test_precision"].mean(),
        "cv_recall": logreg_cv["test_recall"].mean(),
        "cv_f1": logreg_cv["test_f1"].mean(),
        "cv_roc_auc": logreg_cv["test_roc_auc"].mean()
    }

     # Log metrics
    mlflow.log_metrics(logreg_cv_metrics)

    # Log best params
    mlflow.log_params(logreg_gs.best_params_)

    # Log model
    mlflow.sklearn.log_model(
        logreg_gs.best_estimator_,
        artifact_path="logRegModel"
    )

    joblib.dump(logreg_gs.best_estimator_,
                "../../../../../../../Users/Deepak Khatri/Documents/MLOPS_Assignment_1/MODELS/logRegModel.pkl")
    mlflow.log_artifact("logRegModel.pkl")


# In[23]:


# Start MLflow run
mlflow.set_experiment("RandomForest_GridSearch_" + datetime_string)
with mlflow.start_run(run_name="RandomForest_GridSearch_" + datetime_string):
    rf_gs.fit(X, y)

    # Cross-validated evaluation of best model
    rf_cv = cross_validate(
        rf_gs.best_estimator_,
        X, y,
        cv=skf,
        scoring=scoring,
        n_jobs=-1
    )

    # Aggregate metrics
    rf_cv_metrics = {
        "cv_accuracy": rf_cv["test_accuracy"].mean(),
        "cv_precision": rf_cv["test_precision"].mean(),
        "cv_recall": rf_cv["test_recall"].mean(),
        "cv_f1": rf_cv["test_f1"].mean(),
        "cv_roc_auc": rf_cv["test_roc_auc"].mean()
    }

     # Log metrics
    mlflow.log_metrics(rf_cv_metrics)

    # Log best params
    mlflow.log_params(rf_gs.best_params_)

    # Log model
    mlflow.sklearn.log_model(
        rf_gs.best_estimator_,
        artifact_path="rfModel"
    )


# ## 3.6 Final evaluation with stratified cross‑validation
# Report accuracy, precision, recall, ROC‑AUC for each best model using cross_validate. (This evaluates generalization beyond the refit scores.)

# In[24]:


scoring = {
    'accuracy':  'accuracy',
    'precision': 'precision',
    'recall':    'recall',
    'roc_auc':   'roc_auc'
}

# Evaluate best Logistic Regression
logreg_cv = cross_validate(
    logreg_gs.best_estimator_,
    X, y,
    cv=skf,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=False
)

# Evaluate best Random Forest
rf_cv = cross_validate(
    rf_gs.best_estimator_,
    X, y,
    cv=skf,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=False
)

def summarize_cv(name, cvres):
    print(f"\n{name} (5-fold CV)")
    for k in ['test_accuracy','test_precision','test_recall','test_roc_auc']:
        print(f"  {k.replace('test_','').upper():10s}: "
              f"mean={cvres[k].mean():.3f}  std={cvres[k].std():.3f}")

summarize_cv("Logistic Regression", logreg_cv)
summarize_cv("Random Forest",      rf_cv)


# ## 3.7 Cross‑validated ROC curve (optional)
# Plot ROC using out‑of‑fold probabilities from cross_val_predict.

# In[25]:


# Cross-validated probabilities
y_proba_lr = cross_val_predict(
    logreg_gs.best_estimator_, X, y, cv=skf, method='predict_proba', n_jobs=-1
)[:, 1]

y_proba_rf = cross_val_predict(
    rf_gs.best_estimator_, X, y, cv=skf, method='predict_proba', n_jobs=-1
)[:, 1]

# ROC curves
fpr_lr, tpr_lr, _ = roc_curve(y, y_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y, y_proba_rf)
auc_lr = roc_auc_score(y, y_proba_lr)
auc_rf = roc_auc_score(y, y_proba_rf)

plt.figure(figsize=(7,5))
plt.plot(fpr_lr, tpr_lr, label=f'LogReg (AUC={auc_lr:.3f})', lw=2)
plt.plot(fpr_rf, tpr_rf, label=f'RandomForest (AUC={auc_rf:.3f})', lw=2)
plt.plot([0,1],[0,1],'k--',alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Cross-validated ROC Curves')
plt.legend()
plt.grid(alpha=0.2)
plt.show()


# ## 3.8 Inspect feature contributions
# 
# For Random Forest, you can review global feature importances (after one‑hot).
# For Logistic Regression, inspect coefficients per one‑hot feature.

# In[26]:


# RF feature importances
rf_best = rf_gs.best_estimator_
feat_names = rf_best.named_steps['prep'].get_feature_names_out()
importances = rf_best.named_steps['clf'].feature_importances_

fi = pd.DataFrame({'feature': feat_names, 'importance': importances}) \
       .sort_values('importance', ascending=False)
print("\nTop 10 RF features:\n", fi.head(10))

# Logistic coefficients
lr_best = logreg_gs.best_estimator_
coefs = lr_best.named_steps['clf'].coef_.ravel()
lr_df = pd.DataFrame({'feature': feat_names, 'coef': coefs}) \
          .sort_values('coef', ascending=False)
print("\nTop +ve LR features:\n", lr_df.head(10))
print("\nTop -ve LR features:\n", lr_df.tail(10))


# In[27]:


def cv_report(name, cvres):
    metrics = ['test_accuracy','test_precision','test_recall','test_roc_auc']
    row = {m.replace('test_',''): (cvres[m].mean(), cvres[m].std()) for m in metrics}
    print(f"\n{name} CV summary")
    for m,(mu,s) in row.items():
        print(f"{m:9s}: {mu:.3f} ± {s:.3f}")

cv_report("Logistic Regression", logreg_cv)
cv_report("Random Forest",      rf_cv)


# # Step 4: Save best model as a PKL file
# 
# As per evaluation metrics (Accuracy, Precision, Recall, ROC/AUC) above, best model is logistic regression.
# Thus saving logistic regression model as a Pickle file.

# In[28]:


import pickle

# Save model as PKL
with open("../../../../../../../Users/Deepak Khatri/Documents/MLOPS_Assignment_1/MODELS/logreg_model.pkl", "wb") as f:
    pickle.dump(lr_best, f)

print("Best model saved as logreg_model.pkl")
print("Best params:", logreg_gs.best_params_)


# # Sample Prediction of one data sample

# In[29]:


sample_patient = {
    "age": 54,
    "sex": 1,
    "cp": 2,
    "trestbps": 180,
    "chol": 246,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 4.5,
    "slope": 1,
    "ca": 0,
    "thal": 2
}


X_sample = pd.DataFrame([sample_patient])

prediction = lr_best.predict(X_sample)[0]
probability = lr_best.predict_proba(X_sample)[0]
confidence = max(probability)

print(prediction)
print(confidence)


# In[30]:


# import mlflow
# from mlflow.tracking import MlflowClient

# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# client = MlflowClient()

# client.transition_model_version_stage(
#     name="LogisticRegressionModel_V2",
#     version=1,
#     stage="Production"
# )


# In[ ]:




