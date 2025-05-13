import { Component } from '@angular/core';

interface TimelineItem {
  imageSrc: string;
  date: string;
  title: string;
  description: string;
  inverted: boolean;
}



@Component({
  selector: 'app-about',
  templateUrl: './about.component.html',
  styleUrls: ['./about.component.css']
})
export class AboutComponent {

  timelineItems: TimelineItem[] = [
    {
      imageSrc: 'assets/img/about/1.jpg',
      date: '2025',
      title: 'Creation of Smart Metrics',
      description: 'Smart Metrics was founded in 2025 with the ambition to turn complex data into clear and actionable insights, specializing in Business Intelligence and financial analysis.',
      inverted: false
    },
    {
      imageSrc: 'assets/img/about/2.jpg',
      date: 'Mid-2025',
      title: 'Birth of Vision360°',
      description: 'Vision360° was launched to provide a complete and real-time view of financial and operational KPIs, empowering decision-makers with intuitive dashboards and strategic metrics.',
      inverted: true
    },
    {
      imageSrc: 'assets/img/about/3.jpg',
      date: '2026',
      title: 'Expanding Our Expertise',
      description: 'Building on the success of Vision360°, Smart Metrics expanded its services to include anomaly detection, supplier trust analysis, financial KPIs optimization, and customer satisfaction analytics.',
      inverted: false
    },
    {
      imageSrc: 'assets/img/about/4.jpg',
      date: 'Today',
      title: 'Our Commitment',
      description: 'Today, Smart Metrics continues to innovate, delivering precision, speed, and clarity to help organizations unlock the full potential of their data.',
      inverted: true
    }
  ];

}
