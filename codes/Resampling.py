import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Step 1: Load the dataset
file_path = "C:/Users/iftee/OneDrive/Documents/7th Semester/Thesis/Testing/fake_job_postings.csv"
df = pd.read_csv(file_path)

# Step 2: Select important features (modify this list based on your domain knowledge)
important_categorical_features = [
    'employment_type',
    'required_experience',
    'required_education',
    'function',
    'industry',
    'department'
]

important_numeric_features = [
    'telecommuting',
    'has_company_logo',
    'has_questions'
]

# Step 3: Preprocess categorical features to reduce cardinality
def reduce_categories(df, categorical_columns, threshold=50):
    df = df.copy()
    for col in categorical_columns:
        if col in df.columns:
            # Keep only the top categories by frequency
            value_counts = df[col].value_counts()
            top_categories = value_counts[:threshold].index
            df[col] = df[col].apply(lambda x: x if x in top_categories else 'Other')
    return df

# Apply category reduction
X = reduce_categories(df, important_categorical_features)

# Step 4: Select features and target
X = X[important_categorical_features + important_numeric_features]
y = df['fraudulent']

# Step 5: Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, important_numeric_features),
        ('cat', categorical_transformer, important_categorical_features)
    ])

# Step 6: Create the full pipeline
# Adjust SMOTE parameters for memory efficiency
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(sampling_strategy=0.3, random_state=42, k_neighbors=5)),  # Reduced sampling
    ('undersample', RandomUnderSampler(sampling_strategy=0.8, random_state=42))  # Adjusted ratio
])

# Step 7: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print initial class distribution
print("Original training class distribution:", Counter(y_train))

# Step 8: Fit and transform the training data
X_train_resampled, y_train_resampled = full_pipeline.fit_resample(X_train, y_train)
print("Resampled training class distribution:", Counter(y_train_resampled))

# Step 9: Transform test data
preprocessor = full_pipeline.named_steps['preprocessor']
X_test_transformed = preprocessor.transform(X_test)

# Print shapes to verify dimensionality
print("\nShape information:")
print(f"Original training data shape: {X_train.shape}")
print(f"Resampled training data shape: {X_train_resampled.shape}")
print(f"Transformed test data shape: {X_test_transformed.shape}")
