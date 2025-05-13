import { Component } from '@angular/core';

interface PortfolioItem {
  image: string;
  title: string;
  subtitle: string;
}


@Component({
  selector: 'app-portfolio',
  templateUrl: './portfolio.component.html',
  styleUrls: ['./portfolio.component.css']
})
export class PortfolioComponent {
  portfolioItems: PortfolioItem[] = [
    {
      image: 'assets/img/portfolio/11.png',
      title: 'CFO',
      subtitle: 'Chief Financial Officer'
    },
    {
      image: 'assets/img/portfolio/33.jpg',
      title: 'Sales Manager',
      subtitle: 'Sales Manager'
    },
    {
      image: 'assets/img/portfolio/44.jpg',
      title: 'Internal Auditor',
      subtitle: 'Internal Auditor'
    }
  ];

}
