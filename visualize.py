import numpy as np
import matplotlib.pyplot as plt
import os


# content = [line.rstrip('\r\n').split(',') for line in open('inception_resnet.log')]
# for lign in range(len(content)):
#     for element in range(5):
#         element=float(element)
#         print(element)

# print(content)

data = np.genfromtxt('inception_resnet.log',delimiter=',',names=["epoch","loss","acc","val_loss",'val_acc'])

print(data)

plt.plot(data['epoch'],data['loss'])
plt.ylabel('loss')
plt.show()
