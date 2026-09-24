#import the libraries for the classification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import joblib


#########################
# 2. LOAD DATASET
#########################

df = pd.read_csv("books_cleaned.csv")

print("\n================ DATASET =================")
print(df.head())

##############################
# 7. EXPLORATORY DATA ANALYSIS (EDA)
##############################


# 7.1 Number of Books by Category

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="Category"
)

plt.title("Number of Books by Category")
plt.xlabel("Category")
plt.ylabel("Number of Books")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 7.2 Book Ratings Distribution

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="Rating"
)

plt.title("Distribution of Book Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Books")
plt.tight_layout()
plt.show()


# 7.3 Book Prices Distribution

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="Price",
    bins=15,
    kde=True
)

plt.title("Distribution of Book Prices in Indian Rupees")
plt.xlabel("Price (₹)")
plt.ylabel("Number of Books")
plt.tight_layout()
plt.show()


# 7.4 Price by Rating

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="Rating",
    y="Price"
)

plt.title("Book Price by Rating")
plt.xlabel("Rating")
plt.ylabel("Price (₹)")
plt.tight_layout()
plt.show()


# 7.5 Rating by Category

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="Category",
    y="Rating"
)

plt.title("Book Rating by Category")
plt.xlabel("Category")
plt.ylabel("Rating")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


################################
# 8. DATASET INFORMATION AFTER CLEANING
################################

print("\nFinal Dataset Information:")
df.info()


print("\nFinal Statistical Summary:")
print(df.describe())


################################
# 9. CHECK CLASS IMBALANCE
################################

print("\n================ RATING CLASS DISTRIBUTION =================")

print(
    df["Rating"].value_counts().sort_index()
)


plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="Rating"
)

plt.title("Rating Class Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Books")
plt.tight_layout()
plt.show()


################################
# 10. DEFINE FEATURES AND TARGET
################################

X = df[
    [
        "Price",
        "Availability",
        "Category"
    ]
]


# Target

y = df["Rating"]


print("\nFeatures:")
print(X.head())


print("\nTarget:")
print(y.head())


################################
# 11. TRAIN-TEST SPLIT
################################

# Check minimum number of samples in each class

class_counts = y.value_counts()


print("\nRating class counts:")
print(class_counts)


# If every class has at least 2 samples,
# stratified split can be used.

if class_counts.min() >= 2:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

else:

    print("\nWARNING: Some rating classes contain fewer than 2 samples.")
    print("Stratified splitting cannot be used.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )


print("\nTraining data shape:")
print(X_train.shape)


print("\nTesting data shape:")
print(X_test.shape)


################################
# 12. DEFINE NUMERICAL AND CATEGORICAL FEATURES
################################

numerical_features = [
    "Price"
]


categorical_features = [
    "Availability",
    "Category"
]


################################
# 13. PREPROCESSING
################################


# Numerical preprocessing

numerical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# Categorical preprocessing

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


# Combine preprocessing

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


################################
# 14. LOGISTIC REGRESSION MODEL
################################

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ]
)


# Train Logistic Regression

logistic_model.fit(
    X_train,
    y_train
)


# Prediction

y_pred_logistic = logistic_model.predict(
    X_test
)


# Probability prediction

y_probability_logistic = logistic_model.predict_proba(
    X_test
)


################################
# 15. DECISION TREE MODEL
################################

decision_tree_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            DecisionTreeClassifier(
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)


# Train Decision Tree

decision_tree_model.fit(
    X_train,
    y_train
)


# Prediction

y_pred_tree = decision_tree_model.predict(
    X_test
)


# Probability prediction

y_probability_tree = decision_tree_model.predict_proba(
    X_test
)


################################
# 16. RANDOM FOREST MODEL
################################

random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)


# Train Random Forest

random_forest_model.fit(
    X_train,
    y_train
)


# Prediction

y_pred_forest = random_forest_model.predict(
    X_test
)


# Probability prediction

y_probability_forest = random_forest_model.predict_proba(
    X_test
)


################################
# 17. MODEL EVALUATION FUNCTION
################################

def evaluate_model(
    model_name,
    y_test,
    y_pred,
    y_probability,
    model
):

    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )


    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )


    # AUC-ROC
    # Calculate only when more than one class is present
    # in the test data.

    try:

        if len(np.unique(y_test)) > 1:

            auc = roc_auc_score(
                y_test,
                y_probability,
                multi_class="ovr",
                average="weighted",
                labels=model.classes_
            )

        else:

            auc = np.nan

    except ValueError:

        auc = np.nan


    print("\n======================================")
    print(model_name)
    print("======================================")


    print(
        "Accuracy :",
        round(accuracy, 4)
    )


    print(
        "Precision:",
        round(precision, 4)
    )


    print(
        "Recall   :",
        round(recall, 4)
    )


    print(
        "F1 Score :",
        round(f1, 4)
    )


    if pd.isna(auc):

        print("AUC-ROC  : Not available")

    else:

        print(
            "AUC-ROC  :",
            round(auc, 4)
        )


    print("\nClassification Report:")


    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


    return [
        accuracy,
        precision,
        recall,
        f1,
        auc
    ]


################################
# 18. EVALUATE LOGISTIC REGRESSION
################################

logistic_results = evaluate_model(
    "Logistic Regression",
    y_test,
    y_pred_logistic,
    y_probability_logistic,
    logistic_model
)


################################
# 19. EVALUATE DECISION TREE
################################

tree_results = evaluate_model(
    "Decision Tree",
    y_test,
    y_pred_tree,
    y_probability_tree,
    decision_tree_model
)


################################
# 20. EVALUATE RANDOM FOREST
################################

forest_results = evaluate_model(
    "Random Forest",
    y_test,
    y_pred_forest,
    y_probability_forest,
    random_forest_model
)


################################
# 21. MODEL COMPARISON
################################

results = pd.DataFrame(
    [
        logistic_results,
        tree_results,
        forest_results
    ],
    columns=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "AUC-ROC"
    ],
    index=[
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ]
)


print("\n======================================")
print("MODEL COMPARISON")
print("======================================")


print(
    results.round(4)
)


################################
# 22. MODEL COMPARISON BAR CHART
################################

results.plot(
    kind="bar",
    figsize=(10, 6)
)


plt.title("Comparison of Classification Models")
plt.xlabel("Model")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(title="Evaluation Metrics")
plt.tight_layout()
plt.show()


################################
# 23. CONFUSION MATRIX - LOGISTIC REGRESSION
################################

cm_logistic = confusion_matrix(
    y_test,
    y_pred_logistic
)


plt.figure(figsize=(10, 6))


sns.heatmap(
    cm_logistic,
    annot=True,
    fmt="d",
    cmap="Blues"
)


plt.title(
    "Confusion Matrix - Logistic Regression"
)

plt.xlabel(
    "Predicted Rating"
)

plt.ylabel(
    "Actual Rating"
)

plt.tight_layout()
plt.show()


################################
# 24. CONFUSION MATRIX - DECISION TREE
################################

cm_tree = confusion_matrix(
    y_test,
    y_pred_tree
)


plt.figure(figsize=(10, 6))


sns.heatmap(
    cm_tree,
    annot=True,
    fmt="d",
    cmap="Blues"
)


plt.title(
    "Confusion Matrix - Decision Tree"
)

plt.xlabel(
    "Predicted Rating"
)

plt.ylabel(
    "Actual Rating"
)

plt.tight_layout()
plt.show()


################################
# 25. CONFUSION MATRIX - RANDOM FOREST
################################

cm_forest = confusion_matrix(
    y_test,
    y_pred_forest
)


plt.figure(figsize=(10, 6))


sns.heatmap(
    cm_forest,
    annot=True,
    fmt="d",
    cmap="Blues"
)


plt.title(
    "Confusion Matrix - Random Forest"
)

plt.xlabel(
    "Predicted Rating"
)

plt.ylabel(
    "Actual Rating"
)

plt.tight_layout()
plt.show()


################################
# 26. SAVE CLEANED DATASET
################################

df.to_csv(
    "books_data_inr.csv",
    index=False
)


print("\nCleaned dataset saved as:")
print("books_data_inr.csv")


################################
# 27. SAVE MODEL PIPELINE USING JOBLIB
################################

joblib.dump(
    random_forest_model,
    "book_rating_classification_pipeline.pkl"
)


print("\nRandom Forest pipeline saved as:")
print("book_rating_classification_pipeline.pkl")


################################
# 28. FINAL DATASET CHECK
################################

print("\n======================================")
print("FINAL DATASET")
print("======================================")


print(
    df.head(10)
)


print("\nFinal Shape:")
print(
    df.shape
)


print("\nFinal Data Types:")
print(
    df.dtypes
)


print("\nFinal Missing Values:")
print(
    df.isnull().sum()
)