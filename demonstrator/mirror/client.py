from socketIO_client import SocketIO, LoggingNamespace

s = SocketIO('localhost', 3000, LoggingNamespace)
s.emit('new feeling',  ['happy', 'sad'])
print("hello")
