import pandas as pd
from sklearn.linear_model import LinearRegression

print("==============================")
print("   STUDENT MARKS PREDICTION")
print("==============================")

# Load dataset
df = pd.read_csv("students.csv")

# Input features
X = df[["Hours", "Previous_Marks", "Attendance"]]

# Target
y = df["Marks"]

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Get student information
hours = float(input("Enter study hours: "))
previous_marks = float(input("Enter previous exam marks: "))
attendance = float(input("Enter attendance percentage: "))

# Make prediction
student_data = pd.DataFrame({
    "Hours": [hours],
    "Previous_Marks": [previous_marks],
    "Attendance": [attendance]
})

prediction = model.predict(student_data)

predicted_marks = prediction[0]

print(f"\nPredicted marks: {predicted_marks:.2f}")

if predicted_marks >= 90:
    print("Performance: Excellent")
elif predicted_marks >= 75:
    print("Performance: Very Good")
elif predicted_marks >= 60:
    print("Performance: Good")
else:
    print("Performance: Needs Improvement")