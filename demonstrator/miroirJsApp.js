var fs = require('fs');

var Happy = fs.readFileSync('emoji/Happy.png');
var Sad = fs.readFileSync('emoji/Sad.png');
var Neutral = fs.readFileSync('emoji/Neutral.png');
var Angry = fs.readFileSync('emoji/Angry.png');
var Surprised = fs.readFileSync('emoji/Surprised.png');

var http = require('http');
var url = require('url');

var server = http.createServer(function(req, res) {
    var page = url.parse(req.url).pathname;
    console.log(page);
    res.writeHead(200, {"Content-Type": "image/png"});
    if (page == '/Happiness') {
        res.write(Happy);
    }
    else if (page == '/Sadness') {
        res.write(Sad);
    }
    else if (page == '/Neutral') {
        res.write(Neutral);
    }
    else if (page == '/Anger') {
        res.write(Angry);
    }
    else if (page == '/Surprise') {
        res.write(Surprised);
    }
    res.end();
});
server.listen(8080);


//var server = http.createServer(function(req, res) {
//    var page = url.parse(req.url).pathname;
//    console.log(page);
//    response.writeHead(200, {"Content-Type": "image/png"});
//    if (page == '/happy') 
//  {
//     
//     response.write(Happy);
//     response.end();
//  }
//});
//server.listen(8080);//