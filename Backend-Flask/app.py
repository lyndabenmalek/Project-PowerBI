from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import logging

app = Flask(__name__)
CORS(app)  # Autorise les requêtes du frontend Angular

# Chargement du modèle


Regression_Path ='./models/discount_utilization_model.pkl'
model1 = joblib.load(Regression_Path)

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        discount_offered = float(data.get('discountoffered', 0))
        discount_used = float(data.get('discountused', 0))

        if discount_offered < 0 or discount_used < 0:
            return jsonify({'error': 'Les valeurs doivent être positives'}), 400

        input_df = pd.DataFrame([{
            'discountoffered': discount_offered,
            'discountused': discount_used
        }])

        if model1 is not None:
            predicted_effectiveness = model1.predict(input_df)[0]
            predicted_discount_used = discount_offered * predicted_effectiveness

            return jsonify({
                'success': True,
                'predicted_effectiveness': f"{predicted_effectiveness:.4f}",
                'predicted_effectiveness_percent': f"{predicted_effectiveness:.2%}",
                'predicted_discount_used': f"{predicted_discount_used:.2f}"
            })
        else:
            return jsonify({'error': 'Le modèle n\'est pas disponible'}), 500

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
