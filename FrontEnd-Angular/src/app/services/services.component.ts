import { Component } from '@angular/core';

interface Service {
  icon: string;
  title: string;
  description: string;
}



@Component({
  selector: 'app-services',
  templateUrl: './services.component.html',
  styleUrls: ['./services.component.css']
})
export class ServicesComponent {
   services: Service[] = [
    {
      icon: 'fas fa-search fa-4x mb-3',
      title: 'Anomaly Detection',
      description: 'We use advanced algorithms to detect anomalies and inconsistencies in your data, ensuring high data quality and integrity.'
    },
    {
      icon: 'fas fa-chart-line fa-4x mb-3',
      title: 'Financial Balance Analysis',
      description: 'Our services help you analyze financial statements, identifying areas for optimization and ensuring financial stability.'
    },
    {
      icon: 'fas fa-hand-holding-usd fa-4x mb-3',
      title: 'Claim Minimization',
      description: 'We assist in reducing the number of claims by identifying problem areas and recommending solutions to improve customer relations.'
    },
    {
      icon: 'fas fa-smile fa-4x mb-3',
      title: 'Customer Satisfaction',
      description: 'We help measure and improve customer satisfaction, ensuring your business builds lasting and positive customer relationships.'
    }
  ];

}
