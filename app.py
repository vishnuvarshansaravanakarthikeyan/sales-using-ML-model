import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, r2_score

# ================================
# 1. CREATE SAMPLE DATASET
# ================================
np.random.seed(42)

df = pd.DataFrame({
    "Maths": np.random.randint(50, 100, 100),
    "Science": np.random.randint(50, 100, 100),
    "History": np.random.randint(50, 100, 100),
    "English": np.random.randint(50, 100, 100),
    "Geography": np.random.randint(50, 100, 100),
    "Sales": np.random.randint(1000, 5000, 100),
    "Price": np.random.randint(50, 500, 100)
})

# Add target column (classification)
df["Target"] = np.random.randint(0, 2, 100)

# ================================
# 2. DATA CLEANING
# ================================
df.fillna(df.mean(numeric_only=True), inplace=True)

# ================================
# 3. CORRELATION ANALYSIS
# ================================
corr = df.corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True)
plt.title("Correlation Matrix")
plt.show()

# ================================
# 4. SAMPLING (25%)
# ================================
sample_df = df.sample(frac=0.25)
print("Sample Data:\n", sample_df.head())

# ================================
# 5. Z-TEST (Hypothesis Testing)
# ================================
sample = np.random.normal(150, 10, 40)
sample_mean = np.mean(sample)

pop_mean = 140
std = 10

z = (sample_mean - pop_mean) / (std / np.sqrt(40))

print("\nZ-score:", z)

if z > 1.96:
    print("Reject Null Hypothesis")
else:
    print("Accept Null Hypothesis")

# ================================
# 6. NUMPY OPERATIONS
# ================================
arr = df["Sales"].values

print("\nMean:", np.mean(arr))
print("Sum:", np.sum(arr))
print("Product:", np.prod(arr[:5]))  # small subset

reshaped = arr.reshape(-1, 1)
print("Transpose:\n", reshaped.T)

# ================================
# 7. STUDENT ANALYSIS
# ================================
subjects = ["Maths","Science","History","English","Geography"]

print("\nSubject Averages:\n", df[subjects].mean())

print("Top Student Index:", df[subjects].mean(axis=1).idxmax())
print("Lowest Student Index:", df[subjects].mean(axis=1).idxmin())

pass_rate = (df[subjects] >= 60).mean()
print("Pass Rate:\n", pass_rate)

# ================================
# 8. SALES ANALYSIS
# ================================
df["Revenue"] = df["Sales"] * df["Price"]

print("\nTotal Revenue:", df["Revenue"].sum())

print("Top Product Revenue Row:\n", df.loc[df["Revenue"].idxmax()])

print("Average Quantity Sold:", df["Sales"].mean())

# ================================
# 9. LOGISTIC REGRESSION
# ================================
X = df.drop("Target", axis=1)
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nLogistic Regression Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
plt.title("Confusion Matrix")
plt.show()

# ================================
# 10. LINEAR REGRESSION
# ================================
X_lr = df[["Sales"]]
y_lr = df["Revenue"]

lr = LinearRegression()
lr.fit(X_lr, y_lr)

pred = lr.predict(X_lr)

print("\nR2 Score:", r2_score(y_lr, pred))

plt.scatter(X_lr, y_lr)
plt.plot(X_lr, pred)
plt.title("Linear Regression")
plt.show()