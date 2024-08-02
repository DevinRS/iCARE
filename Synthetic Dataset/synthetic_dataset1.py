import pandas as pd
import numpy as np

# X is 100 variables following a normal distribution rnaging from 0 to 1
X = np.random.uniform(0, 0.5, size=50)
X = np.append(X, np.random.uniform(0.5, 1, size=50))

# Assign random labels to the variables either 0 or 1
target = []
for i in range(100):
    target.append(np.random.randint(0, 2))

# Feature 1, if X is less than 0.5 and target is 0 it is below the threshold hyperplane
# if X is less than 0.5 and target is 1 it is above the threshold hyperplane
# if X is greater than 0.5 and target is 0, append noise
# if X is greater than 0.5 and target is 1, append noise
feature_1 = []
for i in range(len(X)):
    if X[i] < 0.5 and target[i] == 0:
        # append random number between 0 and 0.5
        feature_1.append(np.random.uniform(0, 0.5))
    elif X[i] < 0.5 and target[i] == 1:
        feature_1.append(np.random.uniform(0.5, 1))
    elif X[i] > 0.5 and target[i] == 0:
        feature_1.append(np.random.uniform(0, 1))
    elif X[i] > 0.5 and target[i] == 1:
        feature_1.append(np.random.uniform(0, 1))

# Feature 2, if X is more than 0.5 and target is 0 it is below the threshold hyperplane
# if X is more than 0.5 and target is 1 it is above the threshold hyperplane
# if X is less than 0.5 and target is 0, append noise
# if X is less than 0.5 and target is 1, append noise
feature_2 = []
for i in range(len(X)):
    if X[i] > 0.5 and target[i] == 0:
        feature_2.append(np.random.uniform(0, 0.5))
    elif X[i] > 0.5 and target[i] == 1:
        feature_2.append(np.random.uniform(0.5, 1))
    elif X[i] < 0.5 and target[i] == 0:
        feature_2.append(np.random.uniform(0, 1))
    elif X[i] < 0.5 and target[i] == 1:
        feature_2.append(np.random.uniform(0, 1))

# Plot X vs feature_1 with target labels
import matplotlib.pyplot as plt
plt.scatter(X, feature_1, c=target)
plt.xlabel('X')
plt.ylabel('Feature 1')
# draw vertical line at x = 0.5
plt.axvline(x=0.5, color='r', linestyle='--')
plt.show()

# Plot X vs feature_2 with target labels
plt.scatter(X, feature_2, c=target)
plt.xlabel('X')
plt.ylabel('Feature 2')
# draw vertical line at x = 0.5
plt.axvline(x=0.5, color='r', linestyle='--')
plt.show()

# Convert to pandas dataframe
df = pd.DataFrame({'X': X, 'Feature 1': feature_1, 'Feature 2': feature_2, 'Target': target})

# Save to csv
df.to_csv('synthetic_dataset1.csv', index=False)


