from run_analysis import run_analysis_eguided
import pandas as pd
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

src = 'ExperimentData/early_diabetes_normalized.csv'

df = pd.read_csv(src)

starting_features = ["Age", "Gender"]
starting_feature_size = [3, 6, 9, 12, 14, 15]
target = 'class'
iterations = 200

F1_result = []

for size in starting_feature_size:
    print(f"Running analysis on {src} with {size} features")
    print("=====================================")

        # Run the analysis
    global_prediction, icare_prediction, eguided_prediction, y_actual = run_analysis_eguided(df, target, size, static_features=starting_features, iteration=iterations)
    # F1 calculation
    print("Calculating F1 Score")
    global_f1 = f1_score(y_actual, global_prediction, average='micro')
    icare_f1 = f1_score(y_actual, icare_prediction, average='micro')
    eguided_f1 = f1_score(y_actual, eguided_prediction, average='micro')
    f1 = [global_f1, icare_f1, eguided_f1]
    F1_result.append(f1)
    print(f"Global Recommendation: {global_f1}")
    print(f"Icare Recommendation: {icare_f1}")
    print(f"E-guided Recommendation: {eguided_f1}")
    print("=====================================\n")

# Plot the results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # Create a figure with three subplots side by side

# Plot ROC-AUC Score
axes[0].errorbar(starting_feature_size, [x[0] for x in F1_result], label='Global', fmt='o-')
axes[0].errorbar(starting_feature_size, [x[1] for x in F1_result], label='iCARE', fmt='o-')
axes[0].errorbar(starting_feature_size, [x[2] for x in F1_result], label='eGuided', fmt='o-')
axes[0].set_xlabel('Number of Initial Features')
axes[0].set_ylabel('F1 Score')
axes[0].set_title('Early Diabetes Dataset')
axes[0].legend()

src = 'ExperimentData/Heart Disease Dataset - Preprocessed.csv'

df = pd.read_csv(src)

starting_features = ["age", "sex"]
starting_feature_size = [2, 4, 6, 8, 10, 12]
target = 'target'
iterations = 200

F1_result = []

for size in starting_feature_size:
    print(f"Running analysis on {src} with {size} features")
    print("=====================================")

        # Run the analysis
    global_prediction, icare_prediction, eguided_prediction, y_actual = run_analysis_eguided(df, target, size, static_features=starting_features, iteration=iterations)
    # F1 calculation
    print("Calculating F1 Score")
    global_f1 = f1_score(y_actual, global_prediction, average='micro')
    icare_f1 = f1_score(y_actual, icare_prediction, average='micro')
    eguided_f1 = f1_score(y_actual, eguided_prediction, average='micro')
    f1 = [global_f1, icare_f1, eguided_f1]
    F1_result.append(f1)
    print(f"Global Recommendation: {global_f1}")
    print(f"Icare Recommendation: {icare_f1}")
    print(f"E-guided Recommendation: {eguided_f1}")
    print("=====================================\n")

# Plot ROC-AUC Score
axes[1].errorbar(starting_feature_size, [x[0] for x in F1_result], label='Global', fmt='o-')
axes[1].errorbar(starting_feature_size, [x[1] for x in F1_result], label='iCARE', fmt='o-')
axes[1].errorbar(starting_feature_size, [x[2] for x in F1_result], label='eGuided', fmt='o-')
axes[1].set_xlabel('Number of Initial Features')
axes[1].set_ylabel('F1 Score')
axes[1].set_title('Heart Disease Dataset')
axes[1].legend()

src = 'ExperimentData/Heart Failure Prediction Dataset - Preprocessed.csv'

df = pd.read_csv(src)

starting_features = ["Age", "Sex"]
starting_feature_size = [2, 4, 6, 8, 10]
target = 'HeartDisease'
iterations = 200

F1_result = []

for size in starting_feature_size:
    print(f"Running analysis on {src} with {size} features")
    print("=====================================")

        # Run the analysis
    global_prediction, icare_prediction, eguided_prediction, y_actual = run_analysis_eguided(df, target, size, static_features=starting_features, iteration=iterations)
    # F1 calculation
    print("Calculating F1 Score")
    global_f1 = f1_score(y_actual, global_prediction, average='micro')
    icare_f1 = f1_score(y_actual, icare_prediction, average='micro')
    eguided_f1 = f1_score(y_actual, eguided_prediction, average='micro')
    f1 = [global_f1, icare_f1, eguided_f1]
    F1_result.append(f1)
    print(f"Global Recommendation: {global_f1}")
    print(f"Icare Recommendation: {icare_f1}")
    print(f"E-guided Recommendation: {eguided_f1}")
    print("=====================================\n")

# Plot ROC-AUC Score
axes[2].errorbar(starting_feature_size, [x[0] for x in F1_result], label='Global', fmt='o-')
axes[2].errorbar(starting_feature_size, [x[1] for x in F1_result], label='iCARE', fmt='o-')
axes[2].errorbar(starting_feature_size, [x[2] for x in F1_result], label='eGuided', fmt='o-')
axes[2].set_xlabel('Number of Initial Features')
axes[2].set_ylabel('F1 Score')
axes[2].set_title('Heart Failure Dataset')
axes[2].legend()


plt.tight_layout()  # Adjust layout for better spacing
plt.savefig('ExperimentResult/Experiment4_F1Graph.png')


