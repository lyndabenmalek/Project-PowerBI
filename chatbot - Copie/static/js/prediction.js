// Enregistrer ce fichier comme static/js/prediction.js
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const results = document.getElementById('predictionResults');
    const loading = document.getElementById('loading');
    const discountOfferedInput = document.getElementById('discountoffered');
    const discountUsedInput = document.getElementById('discountused');
    const resetButton = document.getElementById('resetButton');
    const resultArea = document.querySelector('.result-area');
    
    // Initialisation du graphique
    
    
    // Utiliser AJAX pour soumettre le formulaire
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Valider les inputs
        if (!validateInputs()) {
            return;
        }
        
        // Afficher l'animation de chargement
        results.style.display = 'none';
        loading.style.display = 'block';
        
        // Récupérer les données du formulaire
        const formData = {
            discountoffered: parseFloat(discountOfferedInput.value),
            discountused: parseFloat(discountUsedInput.value)
        };
        
        // Envoyer la requête AJAX
        fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData),
        })
        .then(response => response.json())
        .then(data => {
            // Cacher l'animation, afficher les résultats
            loading.style.display = 'none';
            results.style.display = 'block';
            
            if (data.success) {
                // Mettre à jour les résultats
                updateResults(data);
                // Mettre à jour le graphique
                updateChart(myChart, formData.discountoffered, formData.discountused, 
                           parseFloat(data.predicted_effectiveness));
                // Ajouter la classe pour le style des résultats
                resultArea.classList.add('has-result');
                // Ajouter une animation
                results.classList.add('animate__animated', 'animate__fadeIn');
            } else {
                // Afficher l'erreur
                displayError(data.error || "Une erreur est survenue");
            }
        })
        .catch(error => {
            console.error('Error:', error);
            loading.style.display = 'none';
            results.style.display = 'block';
            displayError("Une erreur de connexion est survenue");
        });
    });
    
    // Réinitialiser le formulaire et les résultats
    resetButton.addEventListener('click', function() {
        form.reset();
        clearResults();
        resetChart(myChart);
        resultArea.classList.remove('has-result');
    });
    
    // Mettre à jour le graphique en temps réel lors de la saisie
    discountOfferedInput.addEventListener('input', function() {
        if (discountOfferedInput.value && discountUsedInput.value) {
            updateChartRealTime(myChart, 
                parseFloat(discountOfferedInput.value), 
                parseFloat(discountUsedInput.value));
        }
    });
    
    discountUsedInput.addEventListener('input', function() {
        if (discountOfferedInput.value && discountUsedInput.value) {
            updateChartRealTime(myChart, 
                parseFloat(discountOfferedInput.value), 
                parseFloat(discountUsedInput.value));
        }
    });
    
    // Fonctions utilitaires
    function validateInputs() {
        const discountOffered = parseFloat(discountOfferedInput.value);
        const discountUsed = parseFloat(discountUsedInput.value);
        
        if (isNaN(discountOffered) || discountOffered < 0) {
            displayError("La remise offerte doit être un nombre positif");
            return false;
        }
        
        if (isNaN(discountUsed) || discountUsed < 0) {
            displayError("La remise utilisée doit être un nombre positif");
            return false;
        }
        
        return true;
    }
    
    function displayError(message) {
        results.innerHTML = `
            <div class="alert alert-danger" role="alert">
                <i class="fas fa-exclamation-triangle me-2"></i>
                ${message}
            </div>
        `;
    }
    
    function updateResults(data) {
        results.innerHTML = `
            <div class="result-label">Efficacité de la remise:</div>
            <div class="result-value">${data.predicted_effectiveness_percent}</div>
            
            <div class="result-label mt-3">Estimation en valeur:</div>
            <div class="result-value">${data.predicted_discount_used}€</div>
            
            <div class="small text-muted mt-3">
                Prédiction générée le ${new Date().toLocaleString()}
            </div>
        `;
    }
    
    function clearResults() {
        results.innerHTML = `
            <div class="text-muted text-center pt-4">
                Remplissez le formulaire et cliquez sur "Prédire" pour obtenir les résultats
            </div>
        `;
        results.classList.remove('animate__animated', 'animate__fadeIn');
    }
    
   
});