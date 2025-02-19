# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
    Model Type: RandomForest Classifier
    Date: February 19, 2025
    Version: 1.0
    Parameters: n_estimators: 100, random_state: 42
    scikitlearn

## Intended Use
    Predict information based on census data.
    Researchers/Policy makers studying income and demographics

## Training Data
    Source: Census Income Dataset
    Training data used 80% of the data set

## Evaluation Data
    Evaliation data used 20% of the data set

## Metrics
_Please include the metrics used and your model's performance on those metrics._
    Precision: 0.7419
    Recall: 0.6384
    F1: 0.6863

## Ethical Considerations
    Performance across different demographics should be evaluated

## Caveats and Recommendations
    This model is a historical dataset so it might not reflect today
    Recommendation:
        Use the model with a newer dataset
