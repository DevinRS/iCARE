from run_analysis import run_analysis_all_split
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

src = 'ExperimentData\Heart Disease Dataset - Preprocessed.csv'

df = pd.read_csv(src)

starting_features = ['age', 'sex', 'fbs', 'chol', 'trestbps', 'restecg', 'thalach', 'slope', 'oldpeak', 'exang', 'thal', 'ca', 'cp']
starting_feature_size = [2, 4, 6, 8, 10, 12]
target = 'target'

accuracy_result = []
roc_result = []
accuracy_stderr = []
roc_stderr = []
for size in starting_feature_size:
    print(f"Running analysis on {src} with {size} features")
    print("=====================================")

    # Run the analysis
    global_prediction, sls_prediction, lasso_prediction, icare_prediction, eguided_prediction, y_actual = run_analysis_all_split(df, target, size, static_features=starting_features[:size], iteration=10, split=0.2)

    # ROC-AUC calculation
    accuracy_split = []
    roc_split = []
    for split_number in range(len(global_prediction)):

        global_roc = roc_auc_score(y_actual[split_number], global_prediction[split_number])
        sls_roc = roc_auc_score(y_actual[split_number], sls_prediction[split_number])
        lasso_roc = roc_auc_score(y_actual[split_number], lasso_prediction[split_number])
        icare_roc = roc_auc_score(y_actual[split_number], icare_prediction[split_number])
        eguided_prediction_roc = roc_auc_score(y_actual[split_number], eguided_prediction[split_number])
        roc = [global_roc, sls_roc, lasso_roc, icare_roc, eguided_prediction_roc]
        roc_split.append(roc)

        # Change prediction to binary
        global_prediction[split_number] = [1 if x >= 0.5 else 0 for x in global_prediction[split_number]]
        sls_prediction[split_number] = [1 if x >= 0.5 else 0 for x in sls_prediction[split_number]]
        lasso_prediction[split_number] = [1 if x >= 0.5 else 0 for x in lasso_prediction[split_number]]
        icare_prediction[split_number] = [1 if x >= 0.5 else 0 for x in icare_prediction[split_number]]
        eguided_prediction[split_number] = [1 if x >= 0.5 else 0 for x in eguided_prediction[split_number]]

        # Accuracy calculation
        global_accuracy = accuracy_score(y_actual[split_number], global_prediction[split_number])
        sls_accuracy = accuracy_score(y_actual[split_number], sls_prediction[split_number])
        lasso_accuracy = accuracy_score(y_actual[split_number], lasso_prediction[split_number])
        icare_accuracy = accuracy_score(y_actual[split_number], icare_prediction[split_number])
        eguided_accuracy = accuracy_score(y_actual[split_number], eguided_prediction[split_number])

        accuracy = [global_accuracy, sls_accuracy, lasso_accuracy, icare_accuracy, eguided_accuracy]
        accuracy_split.append(accuracy)
    roc_result.append([sum(x)/len(x) for x in zip(*roc_split)])
    accuracy_result.append([sum(x)/len(x) for x in zip(*accuracy_split)])
    roc_stderr.append([pd.Series(x).sem() for x in zip(*roc_split)])
    accuracy_stderr.append([pd.Series(x).sem() for x in zip(*accuracy_split)])
    print("=====================================\n")

for iterations in range(len(roc_result)):
    print(f"Iteration {iterations + 1}")
    print(f"Number of Initial Features: {starting_feature_size[iterations]}")
    print("=====================================")
    print("ROC-AUC Score:")
    print(f"Global: {roc_result[iterations][0]}")
    print(f"SLS: {roc_result[iterations][1]}")
    print(f"Lasso: {roc_result[iterations][2]}")
    print(f"iCARE: {roc_result[iterations][3]}")
    print(f"eGuided: {roc_result[iterations][4]}")
    print("Accuracy Score:")
    print(f"Global: {accuracy_result[iterations][0]}")
    print(f"SLS: {accuracy_result[iterations][1]}")
    print(f"Lasso: {accuracy_result[iterations][2]}")
    print(f"iCARE: {accuracy_result[iterations][3]}")
    print(f"eGuided: {accuracy_result[iterations][4]}")
    print("=====================================\n")

# Plot the results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Create a figure with two subplots side by side

# Plot ROC-AUC Score
axes[0].errorbar(starting_feature_size, [x[0] for x in roc_result], label='Global', fmt='o-', yerr=[x[0] for x in roc_stderr])
axes[0].errorbar(starting_feature_size, [x[1] for x in roc_result], label='SLS', fmt='o-', yerr=[x[1] for x in roc_stderr])
axes[0].errorbar(starting_feature_size, [x[2] for x in roc_result], label='Lasso', fmt='o-', yerr=[x[2] for x in roc_stderr])
axes[0].errorbar(starting_feature_size, [x[3] for x in roc_result], label='iCARE', fmt='o-', yerr=[x[3] for x in roc_stderr])
axes[0].errorbar(starting_feature_size, [x[4] for x in roc_result], label='eGuided', fmt='o-', yerr=[x[4] for x in roc_stderr])
axes[0].set_xlabel('Number of Initial Features')
axes[0].set_ylabel('ROC-AUC Score')
axes[0].set_title('ROC-AUC Score for Different Methods')
axes[0].legend()

# Plot Accuracy Score
axes[1].errorbar(starting_feature_size, [x[0] for x in accuracy_result], label='Global', fmt='o-', yerr=[x[0] for x in accuracy_stderr])
axes[1].errorbar(starting_feature_size, [x[1] for x in accuracy_result], label='SLS', fmt='o-', yerr=[x[1] for x in accuracy_stderr])
axes[1].errorbar(starting_feature_size, [x[2] for x in accuracy_result], label='Lasso', fmt='o-', yerr=[x[2] for x in accuracy_stderr])
axes[1].errorbar(starting_feature_size, [x[3] for x in accuracy_result], label='iCARE', fmt='o-', yerr=[x[3] for x in accuracy_stderr])
axes[1].errorbar(starting_feature_size, [x[4] for x in accuracy_result], label='eGuided', fmt='o-', yerr=[x[4] for x in accuracy_stderr])
axes[1].set_xlabel('Number of Initial Features')
axes[1].set_ylabel('Accuracy Score')
axes[1].set_title('Accuracy Score for Different Methods')
axes[1].legend()

plt.tight_layout()  # Adjust layout for better spacing
plt.savefig('ExperimentResult/Experiment4_3_AccAucGraph.png')


