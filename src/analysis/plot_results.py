import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# MODEL METRICS (FINAL VALUES)
# -----------------------------

classes = ["Normal (0)", "DR (1)", "Confounding (3)"]

# Baseline CNN metrics
baseline_accuracy = 0.98
baseline_precision = [0.94, 1.00, 1.00]
baseline_recall = [0.99, 1.00, 0.97]
baseline_f1 = [0.97, 1.00, 0.98]

# Hybrid CNN + Multifractal metrics
hybrid_accuracy = 0.99
hybrid_precision = [0.95, 1.00, 1.00]
hybrid_recall = [0.99, 1.00, 0.98]
hybrid_f1 = [0.97, 1.00, 0.99]

# -----------------------------
# 1️⃣ ACCURACY COMPARISON
# -----------------------------
plt.figure()
plt.bar(["Baseline CNN", "Hybrid CNN+MF"],
        [baseline_accuracy, hybrid_accuracy])
plt.ylim(0.9, 1.0)
plt.ylabel("Accuracy")
plt.title("Overall Accuracy Comparison")
plt.grid(axis="y")
plt.show()

# -----------------------------
# 2️⃣ RECALL PER CLASS
# -----------------------------
x = np.arange(len(classes))
width = 0.35

plt.figure()
plt.bar(x - width/2, baseline_recall, width, label="Baseline")
plt.bar(x + width/2, hybrid_recall, width, label="Hybrid")
plt.xticks(x, classes)
plt.ylim(0.9, 1.0)
plt.ylabel("Recall")
plt.title("Recall Comparison per Class")
plt.legend()
plt.grid(axis="y")
plt.show()

# -----------------------------
# 3️⃣ PRECISION PER CLASS
# -----------------------------
plt.figure()
plt.bar(x - width/2, baseline_precision, width, label="Baseline")
plt.bar(x + width/2, hybrid_precision, width, label="Hybrid")
plt.xticks(x, classes)
plt.ylim(0.9, 1.0)
plt.ylabel("Precision")
plt.title("Precision Comparison per Class")
plt.legend()
plt.grid(axis="y")
plt.show()

# -----------------------------
# 4️⃣ F1-SCORE PER CLASS
# -----------------------------
plt.figure()
plt.bar(x - width/2, baseline_f1, width, label="Baseline")
plt.bar(x + width/2, hybrid_f1, width, label="Hybrid")
plt.xticks(x, classes)
plt.ylim(0.9, 1.0)
plt.ylabel("F1-score")
plt.title("F1-score Comparison per Class")
plt.legend()
plt.grid(axis="y")
plt.show()
