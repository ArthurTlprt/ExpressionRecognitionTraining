import csv
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import menpo.io as mio
from multiprocessing.dummy import Pool as ThreadPool

def load_image_thread(row):
    row = row.split(',')
    image = Image.open('../Manually_Annotated_Images/'+row[0])
    image = image.resize((49, 49), resample=Image.BILINEAR)
    data = [ np.asarray(image).astype(np.float32)/255,
    np.array([np.float32(row[1]), np.float32(row[2])]),
    np.uint8(row[3])
    ]
    return data
    
def load_image_thread_rotated(row,angle):
    row = row.split(',')
    image = Image.open('../Manually_Annotated_Images/'+row[0])
    image = image.resize((49, 49), resample=Image.BILINEAR)
    image=image.rotate(angle)
    data = [ np.asarray(image).astype(np.float32)/255,
    np.array([np.float32(row[1]), np.float32(row[2])]),
    np.uint8(row[3])
    ]
    print(type(data))
    return data

csv_names = ['Anger','Neutral', 'Happy', 'Sad', 'Surprise', 'Non_Face']

for csv_name in csv_names:
    print(csv_name)
    current_csv_read=open(csv_name,"r")
    lines = [line.rstrip('\r\n') for line in current_csv_read ]
    if len(lines) == 60000:
        print("fichier de 60000 images, ok pour h5")

        m = ThreadPool(12)
        data = m.map(load_image_thread, lines)
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
        print(len(lines))
        ImagesNumber=len(lines)
        liste_degres=[0, 5, -5 , 10, -10]
        data=[]
        current_csv_write=open(csv_name+" (copie)","a")
        c=0
        while (len(lines)+c != 60000):
            
            print(c)
            if c+ImagesNumber<60000:
                print("dans le if...")
                for line in lines:
                    c+=1
                    #print(line)
                    #csvData=[]
                    m = ThreadPool(12)
                    data=data+m.map(load_image_thread_rotated, lines,liste_degres[i])
                    m.close()
                    #with open('Neutral', 'w') as class_file:
                    wr = csv.writer(current_csv_write)
                    wr.writerow(line)
            else:
                print("dans le else...")
                for line in lines[0:c+ImagesNumber-60000]:
                    m = ThreadPool(12)
                    data = data+m.map(load_image_thread_rotated, lines,liste_degres[i])
                    m.close()
                    wr = csv.writer(current_csv_write)
                    wr.writerow(line)
                break
            i=i+1
           
        images = [x[0] for x in data]
        reg = [x[1] for x in data]
        classi = [x[2] for x in data]
        
        g = h5py.File("../classes/training"+
        csv_name+".hdf5", "w")
        g.create_dataset('data', data=images,dtype=np.float32)
        g.create_dataset('label_regression', data=reg,dtype=np.float32)
        g.create_dataset('label_classification', data=classi,dtype=np.uint8)
        g.close()
                
            
            

                
                
                
            

print("done")
