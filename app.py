from flask import Flask, render_template,request
from tensorflow.keras.models import load_model
import joblib


model = load_model('addiction_model.h5')
scaler = joblib.load('scaler.pkl')
app = Flask(__name__)


@app.route('/',methods=['GET', 'POST'])
def hello_world():
    prediction = None
    if request.method == 'POST':
        # Get values from the form
        data = [
            float(request.form.get('Age')),
            float(request.form.get('Gender')),
            float(request.form.get('School_Grade')),
            float(request.form.get('Daily_Usage_Hours')),
            float(request.form.get('Sleep_Hours')),
            float(request.form.get('Academic_Performance')),
            float(request.form.get('Social_Interactions')),
            float(request.form.get('Exercise_Hours')),
            float(request.form.get('Anxiety_Level')),
            float(request.form.get('Depression_Level')),
            float(request.form.get('Self_Esteem')),
            float(request.form.get('Parental_Control')),
            float(request.form.get('Screen_Time_Before_Bed')),
            float(request.form.get('Phone_Checks_Per_Day')),
            float(request.form.get('Apps_Used_Daily')),
            float(request.form.get('Time_on_Social_Media')),
            float(request.form.get('Time_on_Gaming')),
            float(request.form.get('Time_on_Education')),
            float(request.form.get('Phone_Usage_Purpose')),
            float(request.form.get('Family_Communication')),
            float(request.form.get('Weekend_Usage_Hours'))
        ]

        # Scale and predict
        data_scaled = scaler.transform([data])
        prediction = model.predict(data_scaled)[0][0]
        prediction = round(prediction, 2)

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run()
