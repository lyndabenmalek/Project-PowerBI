import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router'; 
 // Importez votre composant de prédiction ici
import { PredictionComponent } from './prediction/prediction.component'; 
import { HomeComponent } from './home/home.component'; 

const routes: Routes = [
  // { path: '', redirectTo: '/prediction', pathMatch: 'full' },
   { path: '', component: HomeComponent },
  
  { path: 'predict', component: PredictionComponent },
   { path: '**', redirectTo: '' }

];

@NgModule({
  // imports: [RouterModule.forRoot(routes)],
  imports: [RouterModule.forRoot(routes, { scrollPositionRestoration: 'enabled' })],
  exports: [RouterModule]
})
export class AppRoutingModule { }
