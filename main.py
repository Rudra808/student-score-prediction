import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("student_data.csv")

X = df[["Hours_Studied"]]
y = df["Exam_Score"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Test prediction
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)

print("Model R² Score:", round(score, 3))

# Take input from user
hours = float(input("Enter number of study hours: "))

input_data = pd.DataFrame([[hours]], columns=["Hours_Studied"])
predicted_score = model.predict(input_data)

print(f"Predicted score for {hours} hours of study: {predicted_score[0]:.2f}")

# Visualization
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Score")
plt.show()