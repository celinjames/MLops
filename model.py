"""
# 
# File          : model.py
# Created       : 25/05/22 2:53 PM
# Author        : Ron Greego
# Version       : v1.0.0
# Description   :
#
"""

from seaborn import PairGrid, scatterplot, kdeplot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd

data = pd.read_csv('USA_Housing.csv')
print(data)
g = PairGrid(data, diag_sharey=False)
g.map_upper(scatterplot, s=15)
g.map_lower(kdeplot)
g.map_diag(kdeplot, lw=2)

x = data.loc[:, 'Avg. Area Income':'Area Population']
x = x.astype(int)

y = data.loc[:, 'Price']
y = y.astype(int)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Initialize regressor
regressor = LinearRegression()

# Begin fitting the training sets
regressor.fit(X_train, y_train)
# Find the score R2-score for both training and test sets
regressor.score(X_train, y_train)
regressor.score(X_test, y_test)

#Save model
pickle.dump(regressor, open('model.pkl', 'wb'))