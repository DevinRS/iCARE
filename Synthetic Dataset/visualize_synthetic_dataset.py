import pandas as pd
import matplotlib.pyplot as plt

# ------------------- Synthetic Dataset #1 -------------------
# Code Description: Visualize the synthetic dataset #1
df_s1 = pd.read_csv('Synthetic Dataset/synthetic_dataset1.csv')

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.scatter(df_s1['X'], df_s1['Feature 1'], c=df_s1['Target'], cmap='bwr')
ax1.set_xlabel('Initial Feature')
ax1.set_ylabel('Added Feature 1')

# draw horizontal lines at X=0.5
ax1.axvline(x=0.5, color='black', linestyle='--')

# for X>0.5, shade the area with light gray
ax1.axvspan(0.5, 1, color='lightgray', alpha=0.5)
ax1.set_title('Synthetic Dataset #1 w/ Feature 1')

ax2.scatter(df_s1['X'], df_s1['Feature 2'], c=df_s1['Target'], cmap='bwr')
ax2.set_xlabel('Initial Feature')
ax2.set_ylabel('Added Feature 2')

# draw horizontal lines at X=0.5
ax2.axvline(x=0.5, color='black', linestyle='--')

# for X<0.5, shade the area with light gray
ax2.axvspan(0, 0.5, color='lightgray', alpha=0.5)
ax2.set_title('Synthetic Dataset #1 w/ Feature 2')

plt.tight_layout()

# save
plt.savefig('Synthetic Dataset/Figures/synthetic_dataset1_visualization.png', dpi=300)
# ------------------- Synthetic Dataset #1 -------------------


# ------------------- Synthetic Dataset #2 -------------------
# Code Description: Visualize the synthetic dataset #2
df_s2 = pd.read_csv('Synthetic Dataset/synthetic_dataset2.csv')

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.scatter(df_s2['X'], df_s2['Feature 1'], c=df_s2['Target'], cmap='bwr')
ax1.set_xlabel('Initial Feature')
ax1.set_ylabel('Added Feature 1')

# draw horizontal lines at X=0.5
ax1.axvline(x=0.5, color='black', linestyle='--')

# for X>0.5, shade the area with light gray
ax1.axvspan(0.5, 1, color='lightgray', alpha=0.5)
ax1.set_title('Synthetic Dataset #2 w/ Feature 1')

ax2.scatter(df_s2['X'], df_s2['Feature 2'], c=df_s2['Target'], cmap='bwr')
ax2.set_xlabel('Initial Feature')
ax2.set_ylabel('Added Feature 2')

# draw horizontal lines at X=0.5
ax2.axvline(x=0.5, color='black', linestyle='--')

# for X<0.5, shade the area with light gray
ax2.axvspan(0, 0.5, color='lightgray', alpha=0.5)
ax2.set_title('Synthetic Dataset #2 w/ Feature 2')

plt.tight_layout()

# save
plt.savefig('Synthetic Dataset/Figures/synthetic_dataset2_visualization.png', dpi=300)
# ------------------- Synthetic Dataset #2 -------------------


# ------------------- Synthetic Dataset #3 -------------------
# Code Description: Visualize the synthetic dataset #3
df_s3 = pd.read_csv('Synthetic Dataset/synthetic_dataset3.csv')

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.scatter(df_s3['X'], df_s3['Feature 1'], c=df_s3['Target'], cmap='bwr')
ax1.set_xlabel('Initial Feature')
ax1.set_ylabel('Added Feature 1')

# draw horizontal lines at X=0.5
ax1.axvline(x=0.7, color='black', linestyle='--')

# for X>0.7, shade the area with light gray
ax1.axvspan(0.7, 1, color='lightgray', alpha=0.5)
ax1.set_title('Synthetic Dataset #2 w/ Feature 1')

# for 0.3<X<0.7, shade the area with light green
ax1.axvspan(0.3, 0.7, color='lightgreen', alpha=0.5)

ax2.scatter(df_s3['X'], df_s3['Feature 2'], c=df_s3['Target'], cmap='bwr')
ax2.set_xlabel('Initial Feature')
ax2.set_ylabel('Added Feature 2')

# draw horizontal lines at X=0.5
ax2.axvline(x=0.3, color='black', linestyle='--')

# for X<0.3, shade the area with light gray
ax2.axvspan(0, 0.3, color='lightgray', alpha=0.5)
ax2.set_title('Synthetic Dataset #2 w/ Feature 2')

# for 0.3<X<0.7, shade the area with light green
ax2.axvspan(0.3, 0.7, color='lightgreen', alpha=0.5)

plt.tight_layout()

# save
plt.savefig('Synthetic Dataset/Figures/synthetic_dataset3_visualization.png', dpi=300)
# ------------------- Synthetic Dataset #3 -------------------


# ------------------- Synthetic Dataset #4 -------------------
# Code Description: Visualize the synthetic dataset #4
df_s4 = pd.read_csv('Synthetic Dataset/synthetic_dataset4.csv')

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.scatter(df_s4['X'], df_s4['Feature 1'], c=df_s4['Target'], cmap='bwr')
ax1.set_xlabel('Initial Feature')
ax1.set_ylabel('Added Feature 1')

ax2.scatter(df_s4['X'], df_s4['Feature 2'], c=df_s4['Target'], cmap='bwr')
ax2.set_xlabel('Initial Feature')
ax2.set_ylabel('Added Feature 2')

plt.tight_layout()

# save
plt.savefig('Synthetic Dataset/Figures/synthetic_dataset4_visualization.png', dpi=300)
# ------------------- Synthetic Dataset #4 -------------------


# ------------------- Synthetic Dataset #5 -------------------
# Code Description: Visualize the synthetic dataset #5
df_s5 = pd.read_csv('Synthetic Dataset/synthetic_dataset5.csv')

# Create a figure with two subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

ax1.scatter(df_s5['X'], df_s5['Feature 1'], c=df_s5['Target'], cmap='bwr')
ax1.set_xlabel('Initial Feature')
ax1.set_ylabel('Added Feature 1')

ax2.scatter(df_s5['X'], df_s5['Feature 2'], c=df_s5['Target'], cmap='bwr')
ax2.set_xlabel('Initial Feature')
ax2.set_ylabel('Added Feature 2')

ax3.scatter(df_s5['X'], df_s5['Feature 3'], c=df_s5['Target'], cmap='bwr')
ax3.set_xlabel('Initial Feature')
ax3.set_ylabel('Added Feature 3')

plt.tight_layout()

# save
plt.savefig('Synthetic Dataset/Figures/synthetic_dataset5_visualization.png', dpi=300)
# ------------------- Synthetic Dataset #5 -------------------