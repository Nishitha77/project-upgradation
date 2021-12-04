from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN
from PIL import Image
# load image from folder
filename = 'Training Data/Fake_Faces/File_10,001.jpg'

def draw_image_with_boxes(filename, result_list):
    # load the image
    data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']
        # create the shape
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        # draw the box
        ax.add_patch(rect)
    # show the plot
    pyplot.show()
    return(x,y,x+width,y+height)
    
detector = MTCNN()
# detect faces in the image
pixels = pyplot.imread(filename)
faces = detector.detect_faces(pixels)
# display faces on the original image
draw_image_with_boxes(filename, faces)