from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

def load_models():
    models = {}
    model_names = ['ct', 'best_dt', 'best_rf', 'best_xgb', 'best_final_xgb']
    
    for model_name in model_names:
        with open(f'{model_name}.pkl', 'rb') as file:
            models[model_name] = pickle.load(file)
    
    return models

loaded_models = load_models()

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict', methods=['POST'])
def predict():
    Year = float(request.form.get('Year'))
    Min_temp = float(request.form.get('Min_temp'))
    Max_temp = float(request.form.get('Max_temp'))
    AvgTemp = float(request.form.get('AvgTemp'))
    Precipitation = float(request.form.get('Precipitation'))
    Pesticides = float(request.form.get('Pesticides'))
    Item = request.form.get('Item')

    features = pd.DataFrame([[Year, Min_temp, Max_temp, AvgTemp, Precipitation, Pesticides, Item]], 
                            columns=['Year', 'Min_temp', 'Max_temp', 'AvgTemp', 'Precipitation', 'Pesticides', 'Item'])
   
    transform_features = loaded_models['ct'].transform(features)
 
    pred_dt = loaded_models['best_dt'].predict(transform_features)
    pred_rf = loaded_models['best_rf'].predict(transform_features)
    pred_xgb = loaded_models['best_xgb'].predict(transform_features)

    stacked_preds = np.column_stack((pred_dt, pred_rf, pred_xgb))
    
    predicted_yield = loaded_models['best_final_xgb'].predict(stacked_preds)
    
    result = {
        'Year': Year,
        'Min_temp': Min_temp,
        'Max_temp': Max_temp,
        'AvgTemp': AvgTemp,
        'Precipitation': Precipitation,
        'Pesticides': Pesticides,
        'Item': Item,
        'Predicted_Yield': float(predicted_yield[0])
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)