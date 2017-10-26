import csv
import numpy as np


csv_data = [line.rstrip('\r\n') for line in open('../training.csv')]
csv_data = csv_data[1:]

Neutral = []
Happy = []
Sad = []
Surprise = []
Anger = []
Non_Face = []


for line in csv_data:
    line = line.split(',')
    data = [line[0], line[7], line[8], line[6]]
    if data[3] == '0' and len(Neutral) < 60000:
        data.append('0')
        Neutral.append(data)
    elif data[3] == '1' and len(Happy) < 60000:
        data.append('0')
        Happy.append(data)
    elif data[3] == '2' and len(Sad) < 60000:
        data.append('1')
        Sad.append(data)
    elif data[3] == '3' and len(Surprise) < 60000:
        data.append('1')
        Surprise.append(data)
    elif data[3] == '5' and len(Anger) < 60000:
        data.append('1')
        Anger.append(data)
    elif data[3] == '10' and len(Non_Face) < 60000:
        data.append('0')
        Non_Face.append(data)


with open('Neutral', 'w') as class_file:
    wr = csv.writer(class_file)
    wr.writerows(Neutral)

with open('Happy', 'w') as class_file:
    wr = csv.writer(class_file)
    wr.writerows(Happy)

with open('Sad', 'w') as class_file:
    wr = csv.writer(class_file)
    wr.writerows(Sad)

with open('Surprise', 'w') as class_file:
    wr = csv.writer(class_file)
    wr.writerows(Surprise)

with open('Anger', 'w') as class_file:
    wr = csv.writer(class_file)
    wr.writerows(Anger)

with open('Non_Face', 'w') as class_file:
    wr = csv.writer(class_file)
    wr.writerows(Non_Face)
