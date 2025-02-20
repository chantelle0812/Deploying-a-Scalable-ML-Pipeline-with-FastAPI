import pytest
# TODO: add necessary import
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import (
    train_model,
    compute_model_metrics,
    inference,
    save_model,
    load_model,
    performance_on_categorical_slice
)

# TODO: implement the first test. Change the function name and input as needed
@pytest.fixture
def sample_data():
    """
    # create sample data for testing
    """
    # Your code here
    df = pd.DataFrame({
        'age': [39, 40, 41, 42],
        'workclass': ['State-gov', 'Private', 'Private', 'Self-emp'],
        'education': ['Bachelors', 'Masters', 'Doctorate', 'Bachelors'],
        'salary': ['<=50K','>50K', '>50k', '<=50K']
    })
    return df

def test_process_data(sample_data):
    """ Test data processing, categorical data correctly encoded"""
    cat_features = ['workclass','education']

    X, y, encoder, lb = process_data(
        sample_data, 
        categorical_features = cat_features, 
        label = 'salary',
        training= True
    )
   # assert len(y) == len(sample_data), f"Expected {len(sample_data)} labels, got {len(y)}"
    assert X.shape[0] == len(sample_data), f"Expected {len(sample_data)} samples, got {X.shape[0]}"
    
    # Check encoder and lb are created
    assert encoder is not None, "Encoder should not be None"
    assert lb is not None, "LabelBinarizer should not be None"
    
    # Check that y is binary
    assert all(isinstance(val, (int, np.integer)) for val in y), "Labels should be integers"
    assert all(val in [0, 1] for val in y), "Labels should be binary (0 or 1)"
    
    # Check X has correct features: 1 numerical + encoded categorical
    expected_features = 1 + len(encoder.get_feature_names_out())  # age + encoded categorical
    assert X.shape[1] == expected_features, f"Expected {expected_features} features, got {X.shape[1]}"

    


# TODO: implement the second test. Change the function name and input as needed
def test_input_validation():
    """
    Test input validation for the API:
    - Tests data types
    - Tests value ranges
    - Tests required fields
    """
    test_input = {
        "age": 39,
        "workclass": "State-gov",
        "education": "Bachelors",
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    
    # Validate input data types
    assert isinstance(test_input["age"], int), "Age should be integer"
    assert isinstance(test_input["workclass"], str), "Workclass should be string"
    assert isinstance(test_input["education"], str), "Education should be string"
    assert isinstance(test_input["hours-per-week"], int), "Hours-per-week should be integer"
    
    # Validate value ranges
    assert 0 <= test_input["age"] <= 100, "Age should be between 0 and 100"
    assert 0 <= test_input["hours-per-week"] <= 168
    


# TODO: implement the third test. Change the function name and input as needed
def test_model_training():
    """
    Test model predictions:
    - Creates sample data
    - Trains model
    - Tests predictions
    """
    # Create sample training data
    X_train = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_train = np.array([0, 1, 1])
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Make predictions
    preds = inference(model, X_train)
    
    # Test predictions
    assert len(preds) == len(y_train), "Number of predictions should match number of samples"
    assert all(isinstance(pred, (int, np.integer)) for pred in preds), "Predictions should be integers"
    assert all(pred in [0, 1] for pred in preds), "Predictions should be binary"

if __name__ == "__main__":
    pytest.main(['-v'])
    
