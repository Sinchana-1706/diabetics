import os
import pickle
import traceback
from dotenv import load_dotenv

import pandas as pd
import markdown
from flask import Flask, render_template, request
from openai import OpenAI

# Load environment variables
load_dotenv()

# Load the trained model
model = pickle.load(open('logistic_model.pkl', 'rb'))

# Hugging Face API setup using OpenAI client
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_API_KEY"),  # Correct usage
)

# Initialize Flask app
app = Flask(__name__)

# Define input fields
FIELDS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

@app.route('/')
def home():
    return render_template('index.html', values={})

@app.route('/predict', methods=['POST'])
def predict():
    values = {field: request.form.get(field, '').strip() for field in FIELDS}

    # Check for missing input
    if any(value == '' for value in values.values()):
        return render_template(
            'index.html',
            values=values,
            prediction_text="⚠️ Please fill in all fields.",
            prediction_class="text-warning",
            recommendation=""
        )

    try:
        # Prepare input for model
        input_data = [float(values[field]) for field in FIELDS]
        final_input = pd.DataFrame([input_data], columns=FIELDS)

        # Make prediction
        prediction = model.predict(final_input)[0]
        result_text = "🔴 The patient is likely to have diabetes." if prediction == 1 else "🟢 The patient is unlikely to have diabetes."
        result_class = "text-danger" if prediction == 1 else "text-success"

        # Create prompt for AI recommendation
        prompt = f"""
You are a medical assistant. Based on the following patient details and prediction, generate a friendly and informative recommendation.

Patient Details:
- Pregnancies: {values['Pregnancies']}
- Glucose: {values['Glucose']}
- Blood Pressure: {values['BloodPressure']}
- Skin Thickness: {values['SkinThickness']}
- Insulin: {values['Insulin']}
- BMI: {values['BMI']}
- Diabetes Pedigree Function: {values['DiabetesPedigreeFunction']}
- Age: {values['Age']}

Prediction: {result_text}

Please provide:
1. A one-line summary about the diabetes risk.
2. 3-point **Diet Plan**
3. 3-point **Exercise Plan**
4. 3-point **Lifestyle Tips**
5. A warm health wish.

Respond in friendly language using markdown.
"""

        # Request AI-generated recommendation
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b:cerebras",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=600
        )

        raw_recommendation = response.choices[0].message.content
        formatted_recommendation = markdown.markdown(raw_recommendation)

        # Render result
        return render_template(
            'index.html',
            values=values,
            prediction_text=result_text,
            prediction_class=result_class,
            recommendation=formatted_recommendation
        )

    except Exception as e:
        print(traceback.format_exc())
        return render_template(
            'index.html',
            values=values,
            prediction_text="❌ Something went wrong!",
            prediction_class="text-danger",
            recommendation=f"<pre>{e}</pre>"
        )

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
