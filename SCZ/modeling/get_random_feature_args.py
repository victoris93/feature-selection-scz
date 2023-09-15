import numpy as np
import pandas as pd

models = {
    "Logistic Regression": "lr",
    "Ridge Classifier": "ridge",
    "Linear Discriminant Analysis" : "lda",
    "Random Forest Classifier" : "rf",
    "Naive Bayes" : "nb",
    "CatBoost Classifier" : "catboost",
    "Gradient Boosting Classifier" : "gbc",
    "Ada Boost Classifier" : "ada",
    "Extra Trees Classifier" : "et",
    "Quadratic Discriminant Analysis" : "qda",
    "Light Gradient Boosting Machine" : "lightgbm",
    "K Neighbors Classifier" : "knn",
    "Decision Tree Classifier" : "dt",
    "Extreme Gradient Boosting" : "xgboost",
    "Dummy Classifier" : "dummy",
    "SVM - Linear Kernel" : "svm",
} 

model_results = pd.read_csv('results/model_results.csv')
best_models = model_results.groupby('N features').apply(lambda x: x.loc[x['Accuracy'].idxmax()])

best_models.to_csv('results/best_models.csv', index=False)
filename = "random_test_args.txt"

with open(filename, "a+") as f:
    for n_features in best_models["N features"]:
        model = best_models[best_models["N features"] == n_features]["Model"].values[0]
        model_label = models[model]
        f.write(f"{model_label} {n_features}\n")

print("Random feature test args saved.")

