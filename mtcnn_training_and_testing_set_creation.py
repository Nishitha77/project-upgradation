from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN
from PIL import Image
import os 

## change the folder name accordingly for training and testing
path = 'Training Data/'

folders = os.listdir(path)
folders = folders[1:] ## [1:] to remove .ds_store folder if it is made automatically otherwise just use folder

## Iterate over the folder and detect and crop faces and save them in respective folder
for subs in folders:
    for files in os.listdir(path+subs):
        try:
            if 'Fake' in path+subs+files and 'jpg' in path+subs+files:
                print(path+subs+'/'+files)
                pixels = pyplot.imread(path+subs+'/'+files)
                faces = detector.detect_faces(pixels)
                coordinates = tuple(faces[0]['box'])
                Image.fromarray(pixels).crop(coordinates).save('training/Fake_Faces/'+files)
            elif 'Real' in path+subs+files and 'jpg' in path+subs+files:
                print(path+subs+'/'+files)
                pixels = pyplot.imread(path+subs+'/'+files)
                faces = detector.detect_faces(pixels)
                coordinates = tuple(faces[0]['box'])
                Image.fromarray(pixels).crop(coordinates).save('training/Real_Faces/'+files)
        except (IndexError or SystemError):
            print('Face Not Found')