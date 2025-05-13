import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  
import json
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS  # Add CORS support
from groq import Groq
from transformers import AutoTokenizer, AutoModel
from faiss import read_index, write_index, IndexFlatL2
from dotenv import load_dotenv
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from stop_words import get_stop_words
import pickle 
import pandas as pd
import logging
 
prix_min = 0.8 # exemple
prix_max = 5000.0  # exemple

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
EMBEDDING_MODEL = "BAAI/bge-m3"
FAISS_INDEX_PATH = "models/faiss_index.index"
EMBEDDINGS_PATH = "models/embeddings.npy"
JSON_DATA_PATH = "data/data.json"
SCALER_PATH = "models/scaler.pkl"
CLUSTER_MODEL_PATH = "models/modele_clustering.pkl" 
Regression_Path='models/discount_utilization_model.pkl'
classification_Path='models/ModeleClassification.pkl'


# Initialiser le client Groq
client = Groq(api_key="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")

# Charger le modèle d'embedding
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
model = AutoModel.from_pretrained(EMBEDDING_MODEL) 


#Model clustering 
clustering_model = joblib.load(CLUSTER_MODEL_PATH)
scaler2 = joblib.load(SCALER_PATH) 
model1 = joblib.load(Regression_Path)
model2 = joblib.load(classification_Path)

# Variable globale pour stocker les données
document_data = []



with open('models\model_components.pkl', 'rb') as f:
    components = pickle.load(f)
    
tfidf = components['tfidf']
tfidf_matrix = components['tfidf_matrix']
scaler = components['scaler']
df = components['df']

# Precompute similarity matrix
similarity_text = cosine_similarity(tfidf_matrix)

def recommander_produits(index_produit: int, top_n: int = 5) -> pd.DataFrame:
    sim = similarity_text[index_produit]
    diff_prix = np.abs(df['Prix_normalise'] - df.at[index_produit, 'Prix_normalise'])
    score_prix = 1 - diff_prix
    score_note = df['Note_normalisee']
    score_final = 0.6 * sim + 0.2 * score_prix + 0.2 * score_note

    df_scores = df.copy()
    df_scores['score_final'] = score_final
    recommandations = (
        df_scores
        .drop(index=index_produit)
        .sort_values(by='score_final', ascending=False)
        .head(top_n)
    )
    return recommandations[['Nom du produit', 'Prix', 'Note', 'score_final']]

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def initialize_rag():
    global document_data

    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(EMBEDDINGS_PATH):
        index = read_index(FAISS_INDEX_PATH)
        embeddings = np.load(EMBEDDINGS_PATH)
        with open(JSON_DATA_PATH) as f:
            document_data = json.load(f)
        return index, embeddings

    # Charger et préparer les données
    with open(JSON_DATA_PATH) as f:
        document_data = json.load(f)
    
    # Créer les textes pour l'embedding
    texts = [
        f"Sujet: {item['sujet']}\nQuestion: {item['contenu']['question']}\nRéponse: {item['contenu']['reponse']}"
        for item in document_data
    ]
    
    # Générer les embeddings
    embeddings = np.vstack([get_embedding(text) for text in texts])
    
    # Créer et sauvegarder l'index FAISS
    index = IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    os.makedirs("models", exist_ok=True)
    write_index(index, FAISS_INDEX_PATH)
    np.save(EMBEDDINGS_PATH, embeddings)
    
    return index, embeddings

# Initialiser RAG au démarrage
faiss_index, document_embeddings = initialize_rag()

def rag_retrieval(query, k=3):
    query_embedding = get_embedding(query)
    distances, indices = faiss_index.search(query_embedding, k)
    return [document_data[i] for i in indices[0]]

def generate_response(prompt, context):
    # Formater le contexte pour le LLM
    context_str = "\n\n".join([
        f"Question: {item['contenu']['question']}\nRéponse: {item['contenu']['reponse']}"
        for item in context
    ])
    
    system_prompt = f"""Vous êtes un chatbot de Monoprix France. 
Ne donnez pas de réponses hors contexte et essayez de vous limiter strictement au contexte fourni.

Informations disponibles :
{context_str}

Question du client : {prompt}"""
    
    completion = client.chat.completions.create(
        model="Llama3-70b-8192",
        messages=[
            {
                "role": "system", 
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": "Réponds de manière concise et précise en français. Si l'information n'existe pas dans le contexte, dis simplement que vous ne pouvez pas répondre."
            }
        ],
        temperature=0.3
    )
    
    return completion.choices[0].message.content

# Simulated user data
users = {
    "cof@example.com": {"password": "cofpass", "role": "cof"},
    "saleManager@example.com": {"password": "saleManagerpass", "role": "saleManager"},
    "auditor@example.com": {"password": "auditorpass", "role": "auditor"},
}

@app.route("/index")
def home():
    return render_template("index.html")   # Ton site principal

@app.route("/chatbot")
def chatbot():
    return render_template("chat.html")   # L'interface du chatbot 
@app.route("/dash")
def dashbod():
    # URL du rapport Power BI que vous avez partagé
    powerbi_embed_url = "https://app.powerbi.com/reportEmbed?reportId=a489d90c-4a78-4fdd-bedc-6995bab25f5d&autoAuth=true&ctid=604f1a96-cbe8-43f8-abbf-f8eaf5d85730"
    return render_template("Dashbord.html")

@app.route("/ML")
def ML():
    # URL du rapport Power BI que vous avez partagé
    #powerbi_embed_url = "https://app.powerbi.com/reportEmbed?reportId=a489d90c-4a78-4fdd-bedc-6995bab25f5d&autoAuth=true&ctid=604f1a96-cbe8-43f8-abbf-f8eaf5d85730"
    return render_template("MLDash.html")

@app.route("/pred")
def pred():
    global predicted_dollars
    global prediction_text 
    predicted_dollars=""
    prediction_text=""
    
    return render_template('predection.html')

@app.route("/")
def sign_in():
    return render_template('sign-in.html')



@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role')  # Retrieve the selected role from the form
        
        user = users.get(email)
        if user and user['password'] == password:
            if role == user['role']:  # Check if selected role matches the user's role
                if role == 'cof':
                    return render_template('index.html')  # Render admin template
                elif role == 'saleManager':
                    return render_template('index.html')  # Render client templa
                elif role == 'auditor':
                    return render_template('index.html')
            else:
                return "Role miss match! Please select the correct role for your account.", 403  # Forbidden
        else:
            return "Invalid email or password. Please try again.", 401  # Unauthorized 
        
        
@app.route("/cluster")
def cluster():
    return render_template('clustering.html')

@app.route("/test")
def test():
    return render_template('testDash.html')

@app.route("/predict_cluster", methods=["POST"])
def predict_cluster():
    data = request.get_json()
    try:
        Note = float(data["Note"])
        Prix = float(data["Prix"])
    except (KeyError, ValueError):
        return jsonify({"error": "Paramètres manquants ou invalides."}), 400

    # 1. Min-max scaling de Prix
    prix_scaled = (Prix - prix_min) / (prix_max - prix_min)

    # 2. Préparer X
    X_raw = np.array([[Note, prix_scaled]])

    # 3. Standardisation
    X_scaled = scaler2.transform(X_raw)

    # 4. Prédiction
    cluster = clustering_model.predict(X_scaled)[0]

    return jsonify({"cluster": int(cluster)})


    return render_template('sign-in.html')  # Render the sign-in page for GET requests



##############System de recommondation : 
# @app.route('/recommendation')
# def recommendation_page():
#     return render_template('rec.html')  # Plus besoin de passer la liste des produits

@app.route("/recom", methods=["POST"])
def recommend():
    try:
        data = request.get_json()
        product_name = data['product_name'].strip().lower()
        top_n = int(data.get('top_n', 5))
        
        # Trouver le produit le plus similaire
        new_tfidf = tfidf.transform([product_name])
        sim_scores = cosine_similarity(new_tfidf, tfidf_matrix).flatten()
        best_match_idx = np.argmax(sim_scores)
        max_similarity = sim_scores[best_match_idx]

        if max_similarity < 0.2:  # Seuil de similarité minimal
            return jsonify({
                'status': 'error',
                'message': 'Aucun produit correspondant trouvé'
            })
        
        recommendations = recommander_produits(best_match_idx, top_n)
        return jsonify({
            'status': 'success',
            'input_product': df.iloc[best_match_idx]['Nom du produit'],
            'recommendations': recommendations.to_dict('records')
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })


#############Modele de classififcation ###########################


@app.route('/classification', methods=['GET', 'POST'])
def classification():
    prediction_result = None
    
    if request.method == 'POST':
        try:
            # Récupérer les données du formulaire
            features = {
                'approval_delay_hours': float(request.form['approval_delay_hours']),
                'carrier_handling_days': float(request.form['carrier_handling_days']),
                'transit_days': float(request.form['transit_days']),
                'total_estimated_days': float(request.form['total_estimated_days']),
                'purchase_day_of_week': int(request.form['purchase_day_of_week']),
                'purchase_month': int(request.form['purchase_month']),
                'purchase_hour': int(request.form['purchase_hour']),
                'purchase_season': int(request.form['purchase_season'])
            }
            
            # Créer le dataframe pour la prédiction
            input_df = pd.DataFrame([features])
            
            # Faire la prédiction
            prediction = int(model2.predict(input_df)[0])
            proba = model2.predict_proba(input_df)[0]
            
            # Préparer le résultat
            prediction_result = {
                'prediction': prediction,
                'prediction_label': 'Retard de livraison' if prediction == 1 else 'Livraison à temps',
                'probability': round(proba[1] * 100, 2),  # Probabilité de retard (classe 1)
                'is_late': prediction == 1
            }
            
        except Exception as e:
            prediction_result = {
                'error': f"Erreur: {str(e)}"
            }
    
    # Rendre le template avec les résultats éventuels
    return render_template('classification.html', result=prediction_result)





    ########Modele Regression Lineaire discount utilization:

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Récupérer les données du formulaire
#     discount_offered = float(request.form['discountoffered'])
#     discount_used = float(request.form['discountused'])

#     # Construire le DataFrame
#     input_df = pd.DataFrame([{
#     'discountoffered':discount_offered,
#     'discountused': discount_used
#     }])

#     # Prédiction
#     predicted_effectiveness = model1.predict(input_df)[0]
#     predicted_discount_used = discount_offered * predicted_effectiveness

#     return render_template(
#         'predection.html',
#         prediction_text = f"Predicted Discount Effectiveness: {predicted_effectiveness:.2%}",
#         predicted_dollars = f"Predicted Discount Used: ${predicted_discount_used:.2f}"
#     )


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_text = ""
    predicted_dollars = ""
    
    if request.method == 'POST':
        try:
            # Récupérer et valider les données du formulaire
            discount_offered = float(request.form.get('discountoffered', 0))
            discount_used = float(request.form.get('discountused', 0))
            
            # Validation des entrées
            if discount_offered < 0 or discount_used < 0:
                return render_template(
                    'predection.html',
                    prediction_text="Erreur: Les valeurs doivent être positives",
                    predicted_dollars=""
                )
            
            # Construire le DataFrame
            input_df = pd.DataFrame([{
                'discountoffered': discount_offered,
                'discountused': discount_used
            }])
            
            # Prédiction avec gestion des erreurs
            if model1 is not None:
                predicted_effectiveness = model1.predict(input_df)[0]
                predicted_discount_used = discount_offered * predicted_effectiveness
                
                # Enregistrer les prédictions pour analyse future
                logger.info(f"Prédiction: {discount_offered}, {discount_used} -> {predicted_effectiveness:.4f}")
                
                prediction_text = f"Prédiction d'efficacité de la remise: {predicted_effectiveness:.2%}"
                predicted_dollars = f"Remise utilisée prévue: {predicted_discount_used:.2f}€"
            else:
                prediction_text = "Erreur: Le modèle n'est pas disponible"
                predicted_dollars = ""
                
        except ValueError as e:
            logger.error(f"Erreur de validation: {str(e)}")
            prediction_text = "Erreur: Veuillez saisir des valeurs numériques valides"
            predicted_dollars = ""
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {str(e)}")
            prediction_text = "Une erreur est survenue lors de la prédiction"
            predicted_dollars = ""
    
    return render_template(
        'predection.html',
        prediction_text=prediction_text,
        predicted_dollars=predicted_dollars
    )

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint pour les prédictions (pour les appels AJAX)"""
    try:
        # Récupérer les données JSON
        data = request.get_json()
        discount_offered = float(data.get('discountoffered', 0))
        discount_used = float(data.get('discountused', 0))
        
        # Validation
        if discount_offered < 0 or discount_used < 0:
            return jsonify({
                'error': 'Les valeurs doivent être positives'
            }), 400
        
        # Prédiction
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
            return jsonify({
                'error': 'Le modèle n\'est pas disponible'
            }), 500
            
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500





@app.route("/chat", methods=["POST"])
def chat():
    if request.method == "POST":
        user_message = request.json["message"]
        
        try:
            # Récupération de contexte
            context = rag_retrieval(user_message)
            
            # Génération de réponse
            response = generate_response(user_message, context)
            
            return jsonify({"response": response})
        except Exception as e:
            return jsonify({"response": f"Une erreur est survenue : {str(e)}"})
    else:
        return Response(status=405)  # Method Not Allowed

if __name__ == "__main__":
    app.run(debug=True)