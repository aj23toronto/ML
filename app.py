import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)

# Train model when server starts
data = pd.read_csv("student_marks.csv")
X = data[['Hours','Attendance']]
y = data['Marks']

model = LinearRegression()
model.fit(X, y)

@app.route('/', methods=['GET','POST'])
def home():
    prediction = None

    if request.method == 'POST':
        hours = float(request.form['hours'])
        attendance = float(request.form['attendance'])

        pred = model.predict([[hours, attendance]])
        pred = max(0, min(100, pred[0]))

        prediction = round(pred,2)

    return render_template("index.html", result=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
