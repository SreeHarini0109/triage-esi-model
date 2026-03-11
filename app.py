from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load Model Setup
MODEL_PATH = "best_esi_model.pkl"
IMPUTER_PATH = "imputer.pkl"
SCALER_PATH = "scaler.pkl"

model = None
imputer = None
scaler = None

try:
    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and preprocessors loaded successfully!")
except Exception as e:
    print(f"Warning: Model or preprocessors not found yet ({e}). Please run tune_models.py first.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not imputer or not scaler:
        return jsonify({'error': 'Model is not trained yet. Run tune_models.py first.'}), 500

    try:
        data = request.json
        features = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
        input_data = [data.get(f, 0.0) for f in features]

        # Structure as DataFrame to feed safely
        df_in = pd.DataFrame([input_data], columns=features)
        
        # Preprocess
        arr_in = imputer.transform(df_in)
        arr_in = scaler.transform(arr_in)
        
        # Predict ESI class (0-indexed model output -> 1-indexed ESI label)
        prediction = model.predict(arr_in)[0]
        
        # Ensure it's treated as scalar
        if hasattr(prediction, 'item'):
            prediction = prediction.item()
            
        esi_level = int(prediction) + 1
        
        return jsonify({'esi_level': esi_level})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    import os

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
