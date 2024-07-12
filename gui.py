# Importing Necessary Libraries
import tkinter as tk
from tkinter import filedialog,messagebox
from tkinter import *
from PIL import Image,ImageTk
import numpy
import numpy as np


import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import tensorflow
from tensorflow.keras.models import load_model

# Loading the Model
from keras.models import load_model

age_gender_model = load_model("Age_Sex_Detection.keras")

# Initializing the GUI

top = tk.Tk()
top.geometry('800x600')
top.title('Age & Gender Detector')
top.configure(background='#CDCDCD')

# Initializing the labels (1 for Age and 1 for Sex)
label1 = Label(top,background='#CDCDCD',font=('arial',15,"bold"))
label2 = Label(top,background='#CDCDCD',font=('arial',15,"bold"))
sign_image = Label(top)


yolov5_model = torch.hub.load('ultralytics/yolov5','yolov5l')

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

def detect_people(image): # Returns vertices of the boxes of the faces
    results = yolov5_model(image)
    detections = results.xyxy[0].cpu().numpy() # Returns coordinates of each detection along with class and their confidences
    boxes = []
    for detection in detections:
        x1,y1,x2,y2,confidence,obj = detection
        if int(obj) == 0: # Since in COCO Dataset, 0 is for person
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            boxes.append((x1,y1,x2-x1,y2-y1))    
    return boxes


def detect_shirt_colour(image,box):
    x,y,w,h = box
    x,y,w,h = x,y+h//2,w,h//4
    person_img = image[y:y+h,x:x+w]
    hsv = cv2.cvtColor(person_img,cv2.COLOR_BGR2HSV)
    mask_white = cv2.inRange(hsv,(0,0,200),(180,30,255))
    mask_black = cv2.inRange(hsv,(0,0,0),(180,255,50))
    any_col = cv2.inRange(hsv,(0,0,0),(255,255,255))
    white_pix_cnt = cv2.countNonZero(mask_white)
    black_pix_cnt = cv2.countNonZero(mask_black)
    tot = cv2.countNonZero(any_col)
    col = max(tot-white_pix_cnt-black_pix_cnt,white_pix_cnt,black_pix_cnt)
    if col == white_pix_cnt:
        return 'White'
    elif col == black_pix_cnt:
        return 'Black'
    return 'Other'

def process_image(image_path):
    image = cv2.imread(image_path)
    boxes = detect_people(image)
    male_count,female_count = 0,0
    ages = []
    cnt = 0
    for box in boxes:
        cnt+=1
        x,y,w,h = box
        person_img = image[y:y+h,x:x+w]
        shirt_colour = detect_shirt_colour(image,box)

        face_img = person_img.copy()
        face_img = cv2.cvtColor(face_img,cv2.COLOR_BGR2RGB)
        face_img = transform(face_img).unsqueeze(0)
        face_img = face_img.permute(0,2,3,1).numpy()
        

        predictions = age_gender_model.predict(face_img)
        gender = 'Male' if predictions[0][0] <= 0.5 else 'Female'
        age = int(predictions[1][0])

        if shirt_colour == 'White':
            age = 23
        elif shirt_colour == 'Black' and len(boxes)>2:
            age = 'Child'


        if gender == "Male":
            male_count+=1
        else:
            female_count+=1
        ages.append(age)
        
        label = f'{gender}, Age: {age}'
        if cnt%2==0:
            cv2.rectangle(image,(x,y),((x+w),(y+h)),(0,255,0),1)
            cv2.putText(image,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,150,0),1)
        else:
            cv2.rectangle(image,(x,y),((x+w),(y+h)),(0,255,0),1)
            cv2.putText(image,label,(x,y+h),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,150,0),1)
        
    
    return image, male_count,female_count,ages


def people(image_path):
    processed_image,male_count,female_count,ages = process_image(image_path)
    print(f'No of Males: {male_count},No of Females: {female_count}')
    output_path = 'processed_'+image_path.split('/')[-1]
    cv2.imwrite(output_path,processed_image)
    t1 = f'No of Male: {male_count}'
    t2 = f'No of Female: {female_count}'
    label1.configure(foreground="#011638",text=t1)
    label2.configure(foreground="#011638",text=t2)
    return processed_image

def show_Detect_button(file_path):
    Detect_b = Button(top,text="Detect Image",command=lambda: people(file_path),padx = 10,pady = 5)
    Detect_b.configure(background="#364156",foreground='white',font = ('arial',10,'bold'))
    Detect_b.place(relx = 0.79,rely = 0.46)


def upload_image():
    global file_path
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text = '')
        label2.configure(text = '')
        show_Detect_button(file_path)
    except:
        pass



upload=Button(top,text="Upload an Image",command=upload_image,padx=10,pady=5)
upload.configure(background="#364156",foreground='white',font=('arial',10,'bold'))
upload.pack(side='bottom',pady=50)
sign_image.pack(side='bottom',expand=True)
label1.pack(side="bottom",expand=True)
label2.pack(side="bottom",expand=True)
heading=Label(top,text="No of Male, Female and their ages",pady=20,font=('arial',20,"bold"))
heading.configure(background="#CDCDCD",foreground="#364156")
heading.pack()
top.mainloop()