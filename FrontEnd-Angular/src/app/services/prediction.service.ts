import { Injectable } from '@angular/core';

import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';




export interface PredictionRequest {
  discountoffered: number;
  discountused: number;
}

export interface PredictionResponse {
  success: boolean;
  predicted_effectiveness: number;
  predicted_effectiveness_percent: string;
  predicted_discount_used: number;
  error?: string;
}



@Injectable({
  providedIn: 'root'
})
export class PredictionService {




   // URL de base de l'API Flask (à ajuster selon configuration)
private apiUrl = 'http://127.0.0.1:5000/api';





  constructor(private http: HttpClient) { }



    /**
   * Vérifie si l'API est en ligne
   */
  checkStatus(): Observable<any> {
    return this.http.get(`${this.apiUrl}/status`);
  }

  /**
   * Envoie une demande de prédiction à l'API Flask
   */
 predictDiscountEffectiveness(data: PredictionRequest): Observable<PredictionResponse> {
  return this.http.post<PredictionResponse>(`${this.apiUrl}/predict`, data);
}


}
