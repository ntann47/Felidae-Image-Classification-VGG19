from flask import Flask, redirect, url_for, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
import os
app = Flask(__name__)
cwd = os.getcwd()
path = os.path.join(cwd, "C:\\Users\\An Ngo\\Desktop\\Neural_D\\thucHanhNeural\\NgoTruongAn_TH_ANN\\static\\images\\")
app.config['IMAGES_FOLDER'] = path
model = load_model('C:\\Users\\An Ngo\\Desktop\\Neural_D\\thucHanhNeural\\NgoTruongAn_TH_ANN\\Felidae_ANN_VGG19.h5')
@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')


def resize_image(img):
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img)
    width, height = image.size
    if width == height:
        image = image.resize((224,224), Image.ANTIALIAS)
    else:
        if width > height:
            left = width/2 - height/2
            right = width/2 + height/2
            top = 0
            bottom = height
            image = image.crop((left,top,right,bottom))
            image = image.resize((224,224), Image.ANTIALIAS)
        else:
            left = 0
            right = width
            top = 0
            bottom = width
            image = image.crop((left,top,right,bottom))
            image = image.resize((224,224), Image.ANTIALIAS)
    numpy_image=np.array(image)
    return numpy_image

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = path + imagefile.filename
    imagefile.save(image_path)
    img_arr = cv2.imread(image_path)
    im = resize_image(img_arr)
    ar = np.asarray(im)
    ar = ar.astype('float32')
    ar /= 255.0
    ar = ar.reshape(-1, 224, 224, 3)
    test_predictions = model.predict(ar)
# get model predictions
    maxnum = np.argmax(test_predictions,axis=1)
    pred_prob = test_predictions.max() * 100
    if maxnum == 0:
        prediction = 'Lion' +' - Conf: '+ '{:.2f}'.format(pred_prob)
    elif maxnum == 1:
        prediction = 'Leopard'+' - Conf: '+ '{:.2f}'.format(pred_prob)
    elif maxnum == 2:
        prediction = 'Cheetah'+' - Conf: '+ '{:.2f}'.format(pred_prob)
    elif maxnum == 3:
        prediction = 'Puma'+' - Conf: '+ '{:.2f}'.format(pred_prob)
    elif maxnum == 4:
        prediction = 'Tiger'+' - Conf: '+ '{:.2f}'.format(pred_prob)
    else :
        prediction = 'Not Recognize'
    return render_template('index.html',img = imagefile.filename, predict =prediction)

    
if __name__ == "__main__":
    app.run(debug=True)

# print(app)