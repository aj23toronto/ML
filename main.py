import pandas as pd
from sklearn.linear_model import LinearRegression
data = pd.read_csv("student_marks.csv")

# input features
X = data[['Hours','Attendance']]
# output
y = data['Marks']

model = LinearRegression()
# train model

model.fit(X,y)
hours = float(input("Enter hours studied: "))
attendance = float(input("Enter attendance percentage: "))
prediction = model.predict([[hours, attendance]])
print("Predicted Marks:", round(prediction[0],2))

