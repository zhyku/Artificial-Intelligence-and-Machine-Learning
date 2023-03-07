import torch
import io
import os
import shutil
from PIL import Image
import cv2
import streamlit as st
import numpy as np
from detect_track import run
from ffmpeg import runFFmpeg, buildFFmpegCommand

content = ''
uploadedfilename = ''
        
#BEGIN MODEL SECTION
@st.cache_resource(ttl=1200)
def download_model_file():
    torch.hub.download_url_to_file('https://onedrive.live.com/download?cid=D851855CC499D6A9&resid=D851855CC499D6A9%21526&authkey=AKiGJLac26dOsBE', 'bdd100Kv3.pt', hash_prefix=None, progress=True)

@st.cache_resource(ttl=7200)
def Load_Yolov5_Model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='bdd100Kv3.pt')  # local model

    # set model parameters
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image
    model.eval()
    
    return model

#Download Model file
download_model_file()

#Load Model
model = Load_Yolov5_Model()
#END MODEL SECTION

def converttomp4(video):
    clip = moviepy.VideoFileClip(video + '.avi')
    clip.write_videofile(video + '.mp4')
    
   
#BEGIN CODE SECTION
def detect_image(imagefile):
    image = cv2.imdecode(np.frombuffer(imagefile, np.uint8), cv2.IMREAD_COLOR)
    image_updated = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_updated)
    results.render()
    for img in results.ims:
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(img)
        img_base64.save(bytes_io, format="jpeg")
    return bytes_io.getvalue()

    results.render()  # updates results.imgs with boxes and labels
    for img in results.ims:
        bytes_io = io.BytesIO()

def detect_video(videofile):
    with open(videofile.name, "wb") as buffer:
            shutil.copyfileobj(videofile, buffer)
    filepath = f'{videofile.name}'

    # Open the video file
    video = cv2.VideoCapture(filepath)
    totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get the video frame dimensions
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    fps = video.get(cv2.CAP_PROP_FPS)

    generatedvideoavifile = "generatedvideo.avi"
    generatedvideomp4file = "generatedvideo.mp4"
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(generatedvideoavifile, fourcc, fps, (frame_width, frame_height),True)
    
    progress_text = "Video Frames being detected."
    wait_ind = st.progress(0, text=progress_text)

    i = 0
    # Loop over each frame
    while video.isOpened():
        i += 1
        wait_ind.progress((i-1)/totalFrames, text=progress_text)
        ret, frame = video.read()
        if ret:
            image_updated = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(image_updated)
            results.render()
            results.save()
            if os.path.exists('runs/detect/exp'):
                detected_frame = cv2.imread('runs/detect/exp/image0.jpg')
                out.write(detected_frame)
                shutil.rmtree("runs/")

            # Break the loop if the "q" key is pressed
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    contents = ''
    if os.path.exists(generatedvideoavifile):
        runFFmpeg(buildFFmpegCommand(generatedvideoavifile,generatedvideomp4file)) # video name with avi format convert to mp4 format
        with open(generatedvideomp4file, "rb") as f:
            contents = f.read()  # file contents could be already fully loaded into RAM

    # Release the video and VideoWriter objects
    video.release()
    out.release()
    buffer.close()
    cv2.destroyAllWindows()
    
    #remove temporary files
    os.remove(filepath)    
    os.remove(generatedvideoavifile)
    os.remove(generatedvideomp4file)
    
    return contents


def track_video(videofile):
    with open(videofile.name, "wb") as buffer:
            shutil.copyfileobj(videofile, buffer)

    generatedvideomp4file = "generatedvideo.mp4"
    
    filepath = f'{videofile.name}'
    with st.spinner('Processing Video. Wait for it...'):
        savePath = run(weights='bdd100Kv3.pt', source=filepath, data='configuration/BDD100K_100.yaml', conf_thres=conf_thres, iou_thres=iou_thres,device='cpu')   
    #print(savePath)
    runFFmpeg(buildFFmpegCommand(savePath,generatedvideomp4file)) # video name with avi format convert to mp4 format
    st.success('Done!') 
    # startfile(savePath)
    extension = filepath.split(".")[1]
    with open(generatedvideomp4file, "rb") as f:
        contents = f.read()  # file contents could be already fully loaded into RAM
    if extension != 'mp4':
        os.remove(filepath)
        #os.remove(filepath.replace(f".{extension}", ".mp4"))
        os.remove(generatedvideomp4file)
    else:
        os.remove(filepath)
    shutil.rmtree("runs/")
    return contents

#END CODE SECTION

#BEGIN UI SECTION
st.title('Object Detection and Tracking by IIITH _:blue[Group4 2023]_')
  
with st.sidebar:
    with st.expander("Expand, Choose File (of type image or video) and Upload", expanded=True):
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            content = uploaded_file.getvalue()
            uploadedfilename = uploaded_file.name
            file_type = uploaded_file.type

        #modelsel = st.radio("Model Selection", ('Yolov5', 'Yolov8'))
        conf_thres = st.slider('Select confidence threshold', 0.0, 1.0, model.conf)
        model.conf = conf_thres
        iou_thres = st.slider('Select IOU threshold', 0.0, 1.0, model.iou)
        model.iou = iou_thres
        task_selection = st.radio("Task Selection", ('Detect', 'Track'), 0)

        if task_selection == 'Detect':
            track_btn = st.button("Detect")
        else:
            track_btn = st.button("Track")
        
        if uploadedfilename != '':
            if 'video' in file_type:
                st.header("Uploaded Video") 
                original_video = st.video(content)
            elif 'image' in file_type:
                st.header("Uploaded Image") 
                original_image = st.image(content)
            else:
                st.header("Invalid File Format")

if track_btn:
    if uploadedfilename != '':
        if 'video' in file_type:
            st.header("Processed Video")
            if task_selection == 'Detect':
                updated_content = detect_video(uploaded_file)
                detV = st.video(updated_content)
            else:
                updated_content = track_video(uploaded_file)
                detV = st.video(updated_content)
        elif 'image' in file_type:
            if task_selection == 'Detect':
                st.header("Processed Image")
                updated_content = detect_image(content)
                detI = st.image(updated_content)
            else:
                st.header("Invalid File Format For Tracking")
        else:
            st.header("Invalid File Format")
#END UI SECTION

