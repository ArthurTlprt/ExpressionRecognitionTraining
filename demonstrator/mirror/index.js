var express = require('express');
var app = require('express')();
var http = require('http').Server(app);
var io = require('socket.io')(http);

app.use(express.static(__dirname + '/public'));

app.get('/', function(req, res){
  res.sendFile(__dirname + '/index.html');
  res.sendFile(__dirname +'/emoji/Anger.png');
});

io.on('connection', function(socket) {
  socket.on('new feeling', function(feelings) {
    socket.broadcast.emit('new', feelings);
    console.log(feelings)
  })
});

http.listen(3000, function(){
  console.log('listening on *:3000');
});
