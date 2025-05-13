Vision360 x Monoprix : améliorer les performances financières et la satisfaction client 
Dans le cadre d’un projet appliqué à Monoprix, j’ai développé Vision360, une solution web intelligente permettant :
✅ Le suivi des indicateurs financiers (dépenses, remises, budgets, litiges...)
 🔍 La détection automatique d’anomalies pour alerter en cas de dérives
 😊 L’amélioration de la satisfaction client via un meilleur pilotage des tickets internes
🌐 Développée avec Angular (frontend) et Flask (backend), Vision360 permet aux décideurs de prendre les bonnes décisions, au bon moment, grâce à une interface intuitive et des outils d’analyse puissants (recherche, filtrage, export Excel...).
🛠️ Les outils utilisés :
Talend pour l'extraction, la transformation et le chargement (ETL) des données, assurant une intégration fluide entre les différentes sources d'information.
Power BI pour la visualisation interactive des données, permettant une analyse approfondie des performances financières et la détection d'anomalies.
Ce projet m’a permis de combiner data, finance et expérience utilisateur, dans un contexte métier concret.

projet/
├── backend-flask/          # Application Flask
│   ├── app.py        # API Flask
│   ├── models/       # Modèles ML
│   └── templates/    # Templates Flask (pour la version web)
└── frontend-aNGULAR/         # Application Angular
    ├── src/          # Code source Angular
    └── ...
Prérequis
Backend (Flask)

Python 3.8+
Flask
Flask-CORS
pandas
scikit-learn

Frontend (Angular)

Node.js 14+
Angular CLI 14+

Installation
1. Backend Flask
bash# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement virtuel
# Sur Windows
venv\Scripts\activate
# Sur macOS/Linux
source venv/bin/activate

# Installer les dépendances
pip install flask flask-cors pandas scikit-learn joblib

# S'assurer que le modèle est disponible
# Le modèle doit être situé dans le dossier 'models/discount_utilization_model.pkl'
2. Frontend Angular
bash# Installer les dépendances
cd frontend
npm install
Exécution
1. Démarrer le backend Flask
bash# Dans le dossier backend
python app.py
# Le serveur Flask démarrera sur http://localhost:5000
2. Démarrer le frontend Angular
bash# Dans le dossier frontend
ng serve
# L'application Angular démarrera sur http://localhost:4200
Vous pouvez maintenant accéder à l'application via http://localhost:4200 dans votre navigateur.
Déploiement en production
Préparation du frontend pour la production
bash# Construire l'application Angular pour la production
ng build --configuration production
Configuration du serveur Flask pour servir le frontend
Vous pouvez configurer Flask pour servir les fichiers statiques générés par Angular :
python# Ajouter dans app.py
import os
from flask import send_from_directory



# Servir l'application Angular
@app.route 

