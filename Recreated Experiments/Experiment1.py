from run_analysis import run_analysis_lw_split
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

src_list = [
    'ExperimentData/synthetic_dataset1.csv',
    'ExperimentData/synthetic_dataset2.csv',
    'ExperimentData/synthetic_dataset3.csv',
    'ExperimentData/synthetic_dataset4.csv',
    'ExperimentData/synthetic_dataset5.csv',
]

accuracy_result = []
roc_result = []

for src in src_list:

    print(f"Running analysis on {src}")
    print("=====================================")

    # Load the synthetic dataset
    df = pd.read_csv(src)

    # Run the analysis
    global_prediction, icare_prediction, global_lw_prediction, icare_lw_prediction, y_actual = run_analysis_lw_split(df, 'Target', 1, static_features=['X'], iteration=100, split=0.2)
    global_roc = 0
    icare_roc = 0
    global_lw_roc = 0
    icare_lw_roc = 0
    global_accuracy = 0
    icare_accuracy = 0
    global_lw_accuracy = 0
    icare_lw_accuracy = 0

    for i in range(10):
        # ROC-AUC calculation
        global_roc += roc_auc_score(y_actual[i], global_prediction[i])
        icare_roc += roc_auc_score(y_actual[i], icare_prediction[i])
        global_lw_roc += roc_auc_score(y_actual[i], global_lw_prediction[i])
        icare_lw_roc += roc_auc_score(y_actual[i], icare_lw_prediction[i])
    global_roc /= 10
    icare_roc /= 10
    global_lw_roc /= 10
    icare_lw_roc /= 10
    print("ROC-AUC:")
    print(f"Global Recommendation: {global_roc}")
    print(f"Icare Recommendation: {icare_roc}")
    print(f"Global Recommendation with LW: {global_lw_roc}")
    print(f"Icare Recommendation with LW: {icare_lw_roc}")
    print("=====================================")
    roc = [global_roc, icare_roc, global_lw_roc, icare_lw_roc]
    roc_result.append(roc)

    # Change prediction to binary
    for i in range(10):
        global_prediction[i] = [1 if x >= 0.5 else 0 for x in global_prediction[i]]
        icare_prediction[i] = [1 if x >= 0.5 else 0 for x in icare_prediction[i]]
        global_lw_prediction[i] = [1 if x >= 0.5 else 0 for x in global_lw_prediction[i]]
        icare_lw_prediction[i] = [1 if x >= 0.5 else 0 for x in icare_lw_prediction[i]]

    # Accuracy calculation
    for i in range(10):
        global_accuracy += accuracy_score(y_actual[i], global_prediction[i])
        icare_accuracy += accuracy_score(y_actual[i], icare_prediction[i])
        global_lw_accuracy += accuracy_score(y_actual[i], global_lw_prediction[i])
        icare_lw_accuracy += accuracy_score(y_actual[i], icare_lw_prediction[i])
    global_accuracy /= 10
    icare_accuracy /= 10
    global_lw_accuracy /= 10
    icare_lw_accuracy /= 10
    print("Accuracy:")
    print(f"Global Recommendation: {global_accuracy}")
    print(f"Icare Recommendation: {icare_accuracy}")
    print(f"Global Recommendation with LW: {global_lw_accuracy}")
    print(f"Icare Recommendation with LW: {icare_lw_accuracy}")
    print("=====================================\n")
    accuracy = [global_accuracy, icare_accuracy, global_lw_accuracy, icare_lw_accuracy]
    accuracy_result.append(accuracy)


import numpy as np
import matplotlib.pyplot as plt

# Convert results to numpy arrays for easier slicing
accuracy_result = np.array(accuracy_result)
roc_result = np.array(roc_result)

# Reorder indices to match (Global, Global LW, iCARE, iCARE LW)
reorder_indices = [0, 2, 1, 3]
accuracy_result = accuracy_result[:, reorder_indices]
roc_result = roc_result[:, reorder_indices]

# Dataset labels
datasets_1_3 = ["Synthetic Dataset 1", "Synthetic Dataset 2", "Synthetic Dataset 3"]
datasets_4_5 = ["Synthetic Dataset 4", "Synthetic Dataset 5"]
x_labels_1_3 = np.arange(len(datasets_1_3))  # X locations for datasets 1-3
x_labels_4_5 = np.arange(len(datasets_4_5))  # X locations for datasets 4-5
bar_labels = ["Global", "Global LW", "iCARE", "iCARE LW"]
colors = ["blue", "red", "green", "orange"]
width = 0.2  # Bar width

# Function to plot grouped bar charts
def plot_results(data, roc_data, dataset_labels, x_labels, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    # Accuracy plot
    for i in range(4):  # 4 models
        axes[0].bar(x_labels + i * width, data[:, i], width, label=bar_labels[i], color=colors[i])
    axes[0].set_xticks(x_labels + width * 1.5)
    axes[0].set_xticklabels(dataset_labels)
    axes[0].set_title(f"{title} - Accuracy")
    axes[0].legend()
    axes[0].set_ylabel("Accuracy")

    # ROC-AUC plot
    for i in range(4):  # 4 models
        axes[1].bar(x_labels + i * width, roc_data[:, i], width, label=bar_labels[i], color=colors[i])
    axes[1].set_xticks(x_labels + width * 1.5)
    axes[1].set_xticklabels(dataset_labels)
    axes[1].set_title(f"{title} - ROC-AUC")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].legend()
    axes[1].set_ylabel("ROC-AUC Score")

    plt.tight_layout()
    plt.savefig(f"ExperimentResult/Experiment1_{title}.png")

# Plot for Synthetic Dataset 1-3
plot_results(accuracy_result[:3], roc_result[:3], datasets_1_3, x_labels_1_3, "Synthetic Dataset 1-3")

# Plot for Synthetic Dataset 4-5
plot_results(accuracy_result[3:], roc_result[3:], datasets_4_5, x_labels_4_5, "Synthetic Dataset 4-5")
