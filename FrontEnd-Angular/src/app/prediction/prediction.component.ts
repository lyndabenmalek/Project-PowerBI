import { Component } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { PredictionService, PredictionResponse } from '../services/prediction.service';


@Component({ 

  selector: 'app-prediction',
  templateUrl: './prediction.component.html',
  styleUrls: ['./prediction.component.css']
})
export class PredictionComponent {
  predictionForm: FormGroup;
  loading = false;
  apiStatus = 'Vérification...';
  result: PredictionResponse | null = null;
  error: string | null = null;

  constructor(
    private fb: FormBuilder,
    private predictionService: PredictionService
  ) {
    this.predictionForm = this.fb.group({
      discountoffered: ['', [Validators.required, Validators.min(0)]],
      discountused: ['', [Validators.required, Validators.min(0)]]
    });
  }

  ngOnInit(): void {
    // Vérifier l'état de l'API au chargement
    // this.checkApiStatus();
  }

  

  onSubmit(): void {
    if (this.predictionForm.invalid) {
      Object.keys(this.predictionForm.controls).forEach(key => {
        const control = this.predictionForm.get(key);
        if (control) control.markAsTouched();
      });
      return;
    }

    this.loading = true;
    this.error = null;
    this.result = null;

    const predictionData = {
      discountoffered: parseFloat(this.predictionForm.value.discountoffered),
      discountused: parseFloat(this.predictionForm.value.discountused)
    };

    this.predictionService.predictDiscountEffectiveness(predictionData).subscribe({
      next: (response) => {
        this.loading = false;
        this.result = response;
      },
      error: (err) => {
        this.loading = false;
        this.error = err.error?.error || 'Une erreur s\'est produite lors de la communication avec le serveur';
        console.error('Erreur API:', err);
      }
    });
  }

  resetForm(): void {
    this.predictionForm.reset();
    this.result = null;
    this.error = null;
  }

}
