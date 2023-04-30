import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Load data
dataframe = pd.read_csv('dataset.csv')

# Define the Bayesian network structure
model = BayesianNetwork([('Symptom_1', 'Disease'),
                       ('Symptom_2', 'Disease'),
                       ('Symptom_3', 'Disease'),
                       ('Symptom_4', 'Disease'),
                       ('Symptom_5', 'Disease'),
                       ('Symptom_6', 'Disease'),
                       ('Symptom_7', 'Disease')])

# Estimate the model parameters from data
estimator = MaximumLikelihoodEstimator(model, data=dataframe)
model.fit(dataframe, estimator=MaximumLikelihoodEstimator)

# Define the evidence and query variables
evidence = {'Symptom_1': 'itching', 'Symptom_2': 'skin_rash', 'Symptom_3': 'nodal_skin_eruptions', 'Symptom_4': 'dischromic _patches'}
query = 'Disease'

# Perform inference using the VariableElimination algorithm
inference = VariableElimination(model)
result = inference.query([query], evidence=evidence)

# Print the results
print(result)
