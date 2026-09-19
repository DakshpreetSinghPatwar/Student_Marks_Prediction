from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load dataset
df = pd.read_csv("students.csv")

# Train model
X = df[["Hours", "Previous_Marks", "Attendance"]]
y = df["Marks"]

model = LinearRegression()
model.fit(X, y)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    performance = None

    if request.method == "POST":
        hours = float(request.form["hours"])
        previous_marks = float(request.form["previous_marks"])
        attendance = float(request.form["attendance"])

        student_data = pd.DataFrame({
            "Hours": [hours],
            "Previous_Marks": [previous_marks],
            "Attendance": [attendance]
        })

        prediction = model.predict(student_data)[0]

        if prediction >= 90:
            performance = "Excellent"
        elif prediction >= 75:
            performance = "Very Good"
        elif prediction >= 60:
            performance = "Good"
        else:
            performance = "Needs Improvement"

    return render_template(
        "index.html",
        prediction=prediction,
        performance=performance
    )


if __name__ == "__main__":
    app.run(debug=True)