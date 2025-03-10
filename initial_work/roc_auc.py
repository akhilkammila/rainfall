import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Set a seed for reproducibility
np.random.seed(0)

# Generate sample data
n_samples = 10
# True binary labels (0 or 1)
y_true = [1,1,1,1,1,0,1,1,1,0]
# Predicted probabilities (scores) from a model (random in this case)
y_scores = [1,0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Example')
plt.legend(loc="lower right")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.show()

print(roc_auc)
print(fpr, tpr, thresholds)