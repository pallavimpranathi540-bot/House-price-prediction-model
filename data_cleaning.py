# ============================================================
# HOUSE PRICE PREDICTION WITH NEW USER INPUT
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("datasets.csv")

print("\nFIRST 5 ROWS")
print("=" * 60)
print(df.head())

print("\nDATASET SHAPE")
print("=" * 60)
print(df.shape)


# ============================================================
# 2. REMOVE DUPLICATES
# ============================================================

print("\nDUPLICATE ROWS BEFORE CLEANING:")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("DUPLICATE ROWS AFTER CLEANING:")
print(df.duplicated().sum())


# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

print("\nMISSING VALUES")
print("=" * 60)
print(df.isnull().sum())


# ============================================================
# 4. REMOVE ROWS WHERE PRICE IS MISSING
# ============================================================

df = df.dropna(subset=["Price"])


# ============================================================
# 5. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("Price", axis=1)
y = df["Price"]


print("\nFEATURE COLUMNS")
print("=" * 60)

for column in X.columns:
    print("-", column)


# ============================================================
# 6. IDENTIFY NUMERICAL AND CATEGORICAL COLUMNS
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


print("\nNUMERICAL FEATURES")
print("=" * 60)
print(numeric_features)

print("\nCATEGORICAL FEATURES")
print("=" * 60)
print(categorical_features)


# ============================================================
# 7. NUMERICAL PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# ============================================================
# 8. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# ============================================================
# 9. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)


# ============================================================
# 10. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTRAINING DATA:")
print(X_train.shape)

print("\nTESTING DATA:")
print(X_test.shape)


# ============================================================
# 11. CREATE MODELS
# ============================================================

models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# ============================================================
# 12. TRAIN AND COMPARE MODELS
# ============================================================

results = []

trained_models = {}

print("\n\nMODEL TRAINING")
print("=" * 70)

for name, model in models.items():

    print("\nTraining:", name)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Predict test data
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = pipeline


# ============================================================
# 13. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

print("\n\nMODEL COMPARISON")
print("=" * 80)

print(results_df.to_string(index=False))


# ============================================================
# 14. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\n\nBEST MODEL")
print("=" * 60)

print("Best Model:", best_model_name)

print(
    "Best R2 Score:",
    results_df.iloc[0]["R2 Score"]
)


# ============================================================
# 15. TEST BEST MODEL
# ============================================================

best_predictions = best_model.predict(X_test)

final_mae = mean_absolute_error(
    y_test,
    best_predictions
)

final_mse = mean_squared_error(
    y_test,
    best_predictions
)

final_rmse = np.sqrt(final_mse)

final_r2 = r2_score(
    y_test,
    best_predictions
)


print("\nFINAL MODEL PERFORMANCE")
print("=" * 60)

print("Model :", best_model_name)
print("MAE   :", final_mae)
print("MSE   :", final_mse)
print("RMSE  :", final_rmse)
print("R2    :", final_r2)


# ============================================================
# 16. ACTUAL VS PREDICTED
# ============================================================

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": best_predictions
})

print("\n\nACTUAL VS PREDICTED")
print("=" * 60)

print(comparison.head(10).to_string(index=False))


# ============================================================
# 17. TAKE NEW PROPERTY INPUT
# ============================================================

print("\n\n")
print("=" * 70)
print("NEW PROPERTY PRICE PREDICTION")
print("=" * 70)

print("\nEnter the details of the property.")
print("The program will predict its price.\n")


new_data = {}


for column in X.columns:

    if column in numeric_features:

        while True:

            try:

                value = float(
                    input(f"Enter {column}: ")
                )

                new_data[column] = value

                break

            except ValueError:

                print(
                    "Please enter a valid number."
                )

    else:

        value = input(
            f"Enter {column}: "
        )

        new_data[column] = value


# ============================================================
# 18. CONVERT USER INPUT INTO DATAFRAME
# ============================================================

new_property = pd.DataFrame(
    [new_data],
    columns=X.columns
)


print("\nYOUR PROPERTY DETAILS")
print("=" * 60)

print(new_property.to_string(index=False))


# ============================================================
# 19. PREDICT NEW PROPERTY PRICE
# ============================================================

new_prediction = best_model.predict(
    new_property
)


# ============================================================
# 20. DISPLAY RESULT
# ============================================================

print("\n\n")
print("=" * 70)
print("PREDICTED HOUSE PRICE")
print("=" * 70)

print(
    f"Predicted Price: ₹{new_prediction[0]:,.2f}"
)

print("=" * 70)