from flask import Flask,render_template,request,flash
from torchvision import models
import torchvision.transforms as transforms
import torch.nn as nn
import io
import json
import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
import os
from werkzeug.utils import secure_filename
import pickle
import torch.nn.functional as F
from PIL import Image
UPLOAD_FOLDER = './saved/'
import base64
# from net3d import Net3D


app = Flask(__name__, template_folder="template")
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

class_map = {0: 'laugh', 1: 'pick', 2: 'pour', 3: 'pullup', 4: 'punch'}

class Rnn3D(nn.Module):

    #define the learnable paramters by calling the respective modules (nn.Conv2d, nn.MaxPool2d etc.)
    def __init__(self):
        super(Rnn3D, self).__init__()

        #calling conv3d module for convolution
        conv1 = nn.Conv3d(in_channels = 16, out_channels = 50, kernel_size = 2, stride = 1)

        #calling MaxPool3d module for max pooling with downsampling of 2
        pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=2)

        conv2 =  nn.Conv3d(in_channels = 50, out_channels = 100, kernel_size = (1, 3, 3), stride = 1)

        pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=2)

        self.feat_extractor=nn.Sequential(conv1,nn.ReLU(),pool1,conv2,nn.ReLU(),pool2)

        self.rnn = nn.LSTM(input_size=5625, hidden_size=128, num_layers=1,batch_first=True)
        self.fc1 = nn.Linear(128, 5)



    def forward(self, x):

        b_z, ts, c, h, w = x.shape

        y = self.feat_extractor(x)

        # reinstating the batchsize and frames
        y = y.view(b_z,ts,-1)
        #output has a size of 8x16x128 - basically we have the output for each frame of each clip.
        outp, (hn, cn) = self.rnn(y)
        # We only need the RNN/LSTM output of the last frame since it incorporates all the frame knowledge
        out = self.fc1(outp[:,-1,:])
        return out

@app.route('/')
def form():
    return render_template('formvideo.html')



import cv2
import numpy as np
def get_frames(filename, n_frames= 1):
    frames = []
    v_cap = cv2.VideoCapture(filename)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_list= np.linspace(0, v_len-1, n_frames+1, dtype=np.int16)
    
    for fn in range(v_len):
        success, frame = v_cap.read()
        if success is False:
            continue
        if (fn in frame_list):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            frames.append(frame)
    v_cap.release()
    return frames, v_len

def rgb_array_to_base64_image(rgb_array):
    # Create a PIL Image from the RGB values
    image = Image.fromarray(np.uint8(rgb_array))
    
    # Convert the image to bytes and then to a base64-encoded string
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode()
    
    return base64_image


@app.route('/data/', methods = ['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        form_data = request.form
        file = request.files['vid']
        img_bytes = file.read()
        filename = secure_filename(file.filename)
        file.seek(0)
        file.save(os.path.join('./saved/', filename))


        frames,_ = get_frames(os.path.join('./saved/', filename),n_frames=16)


        h, w = 128,128
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]
        test_transformer = transforms.Compose([
            transforms.Resize((h,w)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ]) 

        frames_tr = []
        for frame in frames:
            frame = Image.fromarray(frame)
            frame = test_transformer(frame)
            frames_tr.append(frame)
        
       

        if len(frames_tr)>0:
            frames_tr = torch.stack(frames_tr)
        
        fixed_model_3d=Rnn3D()
        fixed_model_3d=torch.load('new_model_torch.sav')
        # fixed_model_3d.load_state_dict(torch.load('new_model_torch.pt'))
        # fixed_model_3d = pickle.load(open('original_model2.sav', 'rb'))
        # fixed_model_3d.device('cpu')
        # # import pickle
        # weights = pickle.load(open('cpumodel.sav','rb'))

        # cnt=0
        # for ijk in fixed_model_3d.parameters():
        #     ijk.data = weights[cnt]
        #     cnt+=1

        outputs = fixed_model_3d(frames_tr[:-1].view(1,16,3,128,128))

        _, y_hat = outputs.max(1)
        predicted_idx = y_hat.item()
        

      
        base64_image = rgb_array_to_base64_image(frames[0])
        return render_template('predict.html', predicted_idx=predicted_idx, class_name=class_map[predicted_idx],frames=base64_image)

   
 
 
app.run(host='localhost', port=3306)
