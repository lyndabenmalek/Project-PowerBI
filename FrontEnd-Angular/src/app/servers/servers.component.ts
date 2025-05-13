import {Component, OnInit} from '@angular/core';

@Component({
  selector: 'app-servers',
  templateUrl: './servers.component.html',
  styleUrls: ['./servers.component.css']
})
export class ServersComponent implements OnInit {
  allowServerCreation = false;
  serverCreationStatus = 'notCreated';
  serverName = 'Server 1';
  serverCreated = false;
  servers = ['Server 2', 'Server 3'];

  constructor() {
    setTimeout(() => {
      this.allowServerCreation = true;
    }, 2000)
  }

  ngOnInit() {
    //console.log('hello');
  }

  onServerCreation() {
    this.servers.push(this.serverName);
    this.serverCreated = true;
    this.serverCreationStatus = 'created';
  }

}
