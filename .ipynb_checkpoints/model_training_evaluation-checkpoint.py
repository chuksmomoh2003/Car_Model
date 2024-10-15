import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import cloudpickle as cp

# Load the cleaned dataset
df = pd.read_csv('processed_hyundi.csv')  # Cleaned data from data_prep.py

# Define the feature engineering function
def feature_engineering(X):
    current_year = 2024

    # Avoid chained assignments by operating on temporary variables and assigning back
    X['car_age'] = current_year - X['year']
    X['mileage_per_year'] = X['mileage'] / X['car_age']

    # Calculate fuel efficiency and handle infinite/NaN values
    fuel_efficiency = X['mpg'] / X['engineSize']
    fuel_efficiency.replace([np.inf, -np.inf], np.nan, inplace=True)
    fuel_efficiency.fillna(fuel_efficiency.median(), inplace=True)

    # Assign cleaned column back to the DataFrame
    X['fuel_efficiency_per_engine_size'] = fuel_efficiency
    
    return X

# Categorical columns to encode
categorical_columns = ['model', 'transmission', 'fuelType']

# Numerical columns (including engineered ones)
numerical_columns = ['car_age', 'mileage_per_year', 'fuel_efficiency_per_engine_size', 'tax', 'engineSize']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns),  # Encode categorical variables
        ('num', 'passthrough', numerical_columns)  # Pass numerical variables through
    ]
)

# Full pipeline including feature engineering, preprocessing, and model
pipeline = Pipeline(steps=[
    ('feature_engineering', FunctionTransformer(feature_engineering, validate=False)),  # Apply feature engineering
    ('preprocessor', preprocessor),  # Apply preprocessing (OneHotEncoder)
    ('model', LGBMRegressor(
    objective='regression',
    metric='mse',
    n_jobs=-1,
    random_state=101,
    n_estimators=546,
    num_leaves=49,
    min_child_samples=2,
    learning_rate=0.01340213629132974,
    colsample_bytree=0.3742613512959107,
    reg_alpha=0.03411846537405745,
    reg_lambda=4.1425597420878315
    ))
])

# Split dataset into features and target
X = df.drop(columns=['price'])
y = df['price']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Train the pipeline
pipeline.fit(X_train, y_train)

# Make predictions on the test set
pred = pipeline.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred)

# Print evaluation metrics
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R-squared: {r2}')

# Save evaluation results to a file
with open('evaluation_results.txt', 'w') as f:
    f.write(f'MAE: {mae}\n')
    f.write(f'MSE: {mse}\n')
    f.write(f'RMSE: {rmse}\n')
    f.write(f'R-squared: {r2}\n')

# Save the trained pipeline (including preprocessing and model) using cloudpickle
with open('car_price_pipeline.pkl', 'wb') as f:
    cp.dump(pipeline, f)


