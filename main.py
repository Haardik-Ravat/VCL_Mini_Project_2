import cv2
import torch
import tempfile
import numpy as np
from PIL import Image
import streamlit as st
import torchvision.transforms as transforms
from models import Rnn3D

# original model
model = Rnn3D()
model=torch.load('new_model_torch.pt')



h, w = 128, 128
mean = [0.43216, 0.394666, 0.37645]
std = [0.22803, 0.22145, 0.216989]
transform = transforms.Compose([
    transforms.Resize((h, w)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
pred_map = {0: 'laugh', 1: 'pick', 2: 'pour', 3: 'pullup', 4: 'punch'}


def predict_video(video_file, model):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    v_cap = cv2.VideoCapture(tfile.name)

    frames = []
    transformed_frames = []
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_list = np.linspace(0, v_len-1, 17, dtype=np.int16)

    for fn in range(v_len):
        success, frame = v_cap.read()
        if success is False:
            continue
        if (fn in frame_list):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    v_cap.release()

    for i in range(len(frames)):
        transformed_frames.append(transform(Image.fromarray(frames[i])))

    transformed_frames = torch.stack(transformed_frames, dim=0)
    transformed_frames = transformed_frames.unsqueeze(0)

    with torch.no_grad():
        predictions = model(transformed_frames)
        predictions = torch.argmax(predictions, dim=1)
        predictions = predictions.numpy()
        predictions = predictions.tolist()

    return frames, predictions[0]

st.title("Action Prediction from Videos")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])

if uploaded_file is not None:
    st.subheader("Prediction")
    if st.button("Predict", key=0):
        frames, prediction = predict_video(uploaded_file, model)
        st.image(frames[0], caption="Frame sample", use_column_width=True)
        st.write("Prediction:", pred_map[prediction])
