import pandas as pd
import numpy as np

traits = ['X4_mean', 'X11_mean', 'X18_mean', 'X26_mean', 'X50_mean', 'X3112_mean']
train_data = pd.read_csv('train.csv')
original_means = train_data[traits].mean()
original_variances = train_data[traits].var()

predict_traits = ['X4', 'X11', 'X18', 'X26', 'X50', 'X3112']
predicted_data = pd.read_csv('predictions.csv')
predicted_means = predicted_data[predict_traits].mean()
predicted_variances = predicted_data[predict_traits].var()

for column in predict_traits:
    predicted_data[column] = ((predicted_data[column] - predicted_means[column]) / np.sqrt(predicted_variances[column])) * np.sqrt(predicted_variances[column]*0.85) + original_means[column+'_mean']

# write predicted_data back to csv
predicted_data.to_csv('predicted_data.csv', index=False)