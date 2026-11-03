"""
Step 1: Explore the dataset
---------------------------
Goal: understand what the data looks like BEFORE training any model.
Run from the project folder:
    .\\venv\\Scripts\\python src\\01_explore.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save plots to files instead of opening windows
import matplotlib.pyplot as plt
import pandas as pd

# Build paths relative to this file, so the script works from any folder
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "creditcard.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# 1. Load the CSV into a DataFrame (a table: rows = transactions, columns = features)
df = pd.read_csv(DATA_FILE)

print("=== 1. SIZE ===")
print(f"Rows (transactions): {df.shape[0]:,}")
print(f"Columns (features):  {df.shape[1]}")

print("\n=== 2. COLUMNS ===")
print(list(df.columns))
# Time   = seconds since the first transaction in the dataset
# V1-V28 = anonymised features (PCA-transformed to protect cardholder privacy)
# Amount = transaction amount
# Class  = the label we predict: 0 = genuine, 1 = fraud

print("\n=== 3. MISSING VALUES ===")
print(f"Total missing cells: {df.isnull().sum().sum()}")

print("\n=== 4. CLASS BALANCE (the key problem) ===")
counts = df["Class"].value_counts()
genuine, fraud = counts[0], counts[1]
print(f"Genuine (0): {genuine:,}")
print(f"Fraud   (1): {fraud:,}")
print(f"Fraud share: {fraud / len(df) * 100:.3f}%")
print(f"A model that ALWAYS says 'genuine' would score "
      f"{genuine / len(df) * 100:.2f}% accuracy -- and catch zero fraud.")

print("\n=== 5. AMOUNT: genuine vs fraud ===")
print(df.groupby("Class")["Amount"].describe().round(2))

# Plot 1: class distribution (log scale, otherwise the fraud bar is invisible)
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["Genuine (0)", "Fraud (1)"], [genuine, fraud], color=["#4C72B0", "orange"])
ax.set_yscale("log")
ax.set_ylabel("Number of transactions (log scale)")
ax.set_title("Class distribution in the credit card dataset")
for i, v in enumerate([genuine, fraud]):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom")
fig.tight_layout()
fig.savefig(RESULTS / "01_class_distribution.png", dpi=150)

# Plot 2: transaction amounts for each class
fig, ax = plt.subplots(figsize=(6, 4))
df.boxplot(column="Amount", by="Class", ax=ax, showfliers=False)
ax.set_title("Transaction amount by class (outliers hidden)")
ax.set_xlabel("Class (0 = genuine, 1 = fraud)")
ax.set_ylabel("Amount")
plt.suptitle("")
fig.tight_layout()
fig.savefig(RESULTS / "01_amount_by_class.png", dpi=150)

print(f"\nPlots saved in: {RESULTS}")
