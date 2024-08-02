import pandas as pd
import numpy as np

# X is 100 variables following a normal distribution rnaging from 0 to 1
X = np.random.uniform(0, 0.5, size=50)
X = np.append(X, np.random.uniform(0.5, 1, size=50))

# Assign random labels to the variables either 0 or 1
target = []
for i in range(100):
    target.append(np.random.randint(0, 2))

# Feature 1, if X is greater than 0.7 append random noise
# if X is less than 0.7 and target is 0 it is above the threshold hyperplane
# if X is less than 0.7 and target is 1 it is below the threshold hyperplane
feature_1 = []
for i in range(len(X)):
    if X[i] > 0.7:
        feature_1.append(np.random.uniform(0, 1))
    elif X[i] <= 0.7 and target[i] == 0:
        #90% chance of being 1
        feature_1.append(np.random.choice([np.random.uniform(0, 0.5), np.random.uniform(0.5, 1)], p=[0.1, 0.9]))
    elif X[i] <= 0.7 and target[i] == 1:
        #90% chance of being 0
        feature_1.append(np.random.choice([np.random.uniform(0, 0.5), np.random.uniform(0.5, 1)], p=[0.9, 0.1]))

# Feature 2, is also like feature 1
feature_2 = []
for i in range(len(X)):
    if X[i] > 0.7:
        feature_2.append(np.random.uniform(0, 1))
    elif X[i] <= 0.7 and target[i] == 0:
        #90% chance of being 1
        feature_2.append(np.random.choice([np.random.uniform(0, 0.5), np.random.uniform(0.5, 1)], p=[0.1, 0.9]))
    elif X[i] <= 0.7 and target[i] == 1:
        #90% chance of being 0
        feature_2.append(np.random.choice([np.random.uniform(0, 0.5), np.random.uniform(0.5, 1)], p=[0.9, 0.1]))

# Plot X vs feature_1 with target labels
import matplotlib.pyplot as plt
plt.scatter(X, feature_1, c=target)
plt.xlabel('X')
plt.ylabel('Feature 1')
plt.show()

# Plot X vs feature_2 with target labels
plt.scatter(X, feature_2, c=target)
plt.xlabel('X')
plt.ylabel('Feature 2')
plt.show()


# Convert to pandas dataframe
df = pd.DataFrame({'X': X, 'Feature 1': feature_1, 'Feature 2': feature_2, 'Target': target})

# Save to csv
df.to_csv('synthetic_dataset4.csv', index=False)


