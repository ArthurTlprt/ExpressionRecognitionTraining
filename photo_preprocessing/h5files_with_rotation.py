import csv
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import menpo.io as mio
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial



def load_image_thread(row):
    row = row.split(',')
    try:
        image = Image.open('../Manually_Annotated_Images/'+row[0])
    except IOError:
        print("Image " + row[0] + "not found")
    print("image access")
    image = image.resize((49, 49), resample=Image.BILINEAR)
    data = [ np.asarray(image).astype(np.float32)/255,
    np.array([np.float32(row[1]), np.float32(row[2])]),
    np.uint8(row[3])
    ]
    return data

def load_image_thread_rotated(row, angle):

    row = row.split(',')
    #print('ceci est la ligne',row)
    liste_degres=[0, 5, -5 , 10, -10]
    image = Image.open('../Manually_Annotated_Images/'+row[0])
    image = image.resize((49, 49), resample=Image.BILINEAR)
    image=image.rotate(angle)
    data = [ np.asarray(image).astype(np.float32)/255,
    np.array([np.float32(row[1]), np.float32(row[2])]),
    np.uint8(row[3])
    ]
    return data

csv_names = ['Neutral','Happy','Sad','Anger','Surprise']

for csv_name in csv_names:
    print(csv_name)
    current_csv_read=open(csv_name,"r")
    lines = [line.rstrip('\r\n') for line in current_csv_read ]
    if len(lines) == 60000:
        print("fichier de 60000 images, ok pour h5")

        m = ThreadPool(12)
        data = m.map(load_image_thread, lines)
        print(np.shape(data))
        m.close()


        images = [x[0] for x in data]
        reg = [x[1] for x in data]
        classi = [x[2] for x in data]

        g = h5py.File("../classes/training"+csv_name+".hdf5", "w")
        g.create_dataset('data', data=images,dtype=np.float32)
        g.create_dataset('label_regression', data=reg,dtype=np.float32)
        g.create_dataset('label_classification', data=classi,dtype=np.uint8)
        g.close()
    else:
        i=0
        #print(len(lines))
        ImagesNumber=len(lines)
        liste_degres=[0, -3, -5 , -8, -10]
        data=[]
        current_csv_write=open(csv_name+" (copie)","a")
        print(current_csv_write)
        c=0
        images = []
        reg = []
        classi = []
        angles = []
        while (c != 60000):
            print(c,"/60000")
            if c+ImagesNumber<=60000:
                print("dans le if...")



                #csvData=[]
                m = ThreadPool(12)
                data=m.map(partial(load_image_thread_rotated,angle=liste_degres[i]), lines)#,liste_degres[i])

                m.close()
                images = images+[x[0] for x in data]
                reg = reg+[x[1] for x in data]
                classi = classi+[x[2] for x in data]
                #with open('Neutral', 'w') as class_file:
                with open(csv_name+" (copie)","a") as current_csv_write:
                    for line in lines:
                        wr = csv.writer(current_csv_write)
                        wr.writerow(line.split(','))
                        c+=1
            else:
                print("dans le else...",c)

                #print(c)
                m = ThreadPool(12)
                data=m.map(partial(load_image_thread_rotated,angle=liste_degres[i]), lines[0:c-60000])

                m.close()
                images = images+[x[0] for x in data]
                reg = reg+[x[1] for x in data]
                classi = classi+[x[2] for x in data]
                with open(csv_name+" (copie)","a") as current_csv_write:
                     for line in lines[0:c-60000]:
                        wr = csv.writer(current_csv_write)
                        wr.writerow(line.split(','))



                break

            i=i+1


        g = h5py.File("../classes/training"+
        csv_name+".hdf5", "w")
        g.create_dataset('data', data=images,dtype=np.float32)
        g.create_dataset('label_regression', data=reg,dtype=np.float32)
        g.create_dataset('label_classification', data=classi,dtype=np.uint8)
        g.close()









print("done")
