from tkinter import messagebox
import tkinter as tk
from tkinter import *
from tkinter import filedialog, ttk
from tkinter.filedialog import askopenfilename
import numpy as np 
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO

# Global variables
global filename, person_model, weapon_model
person_labels = ['Person']  # Focusing on 'Person' for YOLOv5s; 'Crowd' needs custom model
weapon_classes = ['knife', 'gun', 'rifle', 'Weapon']  # YOLOv9 classes
CONFIDENCE_THRESHOLD = 0.3
GREEN = (0, 255, 0)  # For person detection
RED = (0, 0, 255)    # For weapon detection

# Function definitions
def graph():
    update_status("Loading training graph...")
    graph_img = cv2.imread('yolov5_model/results.png')
    if graph_img is None:
        text.insert(END, "Failed to load graph image. Ensure 'yolov5_model/results.png' exists.\n")
        update_status("Error loading graph")
        return
    graph_img = cv2.resize(graph_img, (800, 600))
    cv2.imshow("Yolo Training Graph", graph_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    update_status("Ready")

def loadModel():
    global person_model, weapon_model
    text.delete('1.0', END)
    update_status("Loading models...")
    # Load person detection model (YOLOv5s pretrained)
    try:
        person_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        text.insert(END, "Person Detection (YOLOv5s) Model Loaded\n")
    except Exception as e:
        text.insert(END, f"Error loading YOLOv5s model: {str(e)}\n")
        person_model = None

    # Load weapon detection model (YOLOv9 custom .pt file)
    weapon_model_path = 'best.pt'  # Replace with actual path if not in same directory
    try:
        weapon_model = YOLO(weapon_model_path)  # Ultralytics YOLO API
        text.insert(END, "Weapon Detection (YOLOv9) Model Loaded\n")
    except Exception as e:
        text.insert(END, f"Error loading YOLOv9 model: {str(e)}\n")
        weapon_model = None
    update_status("Models loaded" if person_model and weapon_model else "Error loading models")

def imageDetection():
    global person_model, weapon_model
    text.delete('1.0', END)
    update_status("Selecting image...")
    filename = filedialog.askopenfilename(initialdir="images", title="Select Image",
                                          filetypes=(("jpeg files", "*.jpg"), ("png files", "*.png"), ("All files", "*.*")))
    if not filename:
        text.insert(END, "No image file selected. Please select a file.\n\n")
        update_status("No image selected")
        return

    update_status("Processing image...")
    image = cv2.imread(filename)
    if image is None:
        text.insert(END, "Failed to load image. Please try again.\n\n")
        update_status("Error loading image")
        return
    
    # Resize image if it is too large for display
    screen_width = main.winfo_screenwidth()
    screen_height = main.winfo_screenheight()
    image_height, image_width, _ = image.shape
    text.insert(END, f"Original Image Shape: {image_height}x{image_width}\n")
    
    max_display_height = screen_height - 100
    max_display_width = screen_width - 100
    
    if image_height > max_display_height or image_width > max_display_width:
        scaling_factor = min(max_display_height / image_height, max_display_width / image_width)
        new_width = int(image_width * scaling_factor)
        new_height = int(image_height * scaling_factor)
        image = cv2.resize(image, (new_width, new_height))
        text.insert(END, f"Resized Image Shape: {new_height}x{new_width}\n")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Person detection (YOLOv5s)
    person_count = 0
    if person_model is not None:
        results = person_model(image_rgb)
        detections = results.xyxy[0].numpy()  # Bounding boxes
        for det in detections:
            if int(det[5]) == 0:  # Class 0 is 'Person'
                xmin, ymin, xmax, ymax = map(int, det[:4])
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), GREEN, 2)
                cv2.putText(image, 'Person', (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, GREEN, 2)
                person_count += 1
    
    # Weapon detection (YOLOv9)
    weapon_count = 0
    if weapon_model is not None:
        # Preprocess for YOLOv9 (Ultralytics API)
        img = cv2.resize(image_rgb, (640, 640))  # Resize to model input size
        results = weapon_model(img)  # Ultralytics YOLO inference
        detections = results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, conf, cls]
        
        # Rescale boxes to original image size
        scale_x = image_width / 640
        scale_y = image_height / 640
        for det in detections:
            if len(det) >= 6 and det[4] >= CONFIDENCE_THRESHOLD:
                xmin, ymin, xmax, ymax, conf, cls = det[:6]
                cls = int(cls)
                if cls < len(weapon_classes):
                    # Rescale coordinates
                    xmin, xmax = map(int, [xmin * scale_x, xmax * scale_x])
                    ymin, ymax = map(int, [ymin * scale_y, ymax * scale_y])
                    label = weapon_classes[cls]
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), RED, 2)
                    cv2.putText(image, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2)
                    weapon_count += 1
    
    # Display counts
    cv2.putText(image, f"Person Count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2)
    cv2.putText(image, f"Weapon Count: {weapon_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
    
    cv2.imshow("Person and Weapon Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    update_status("Image processing complete")

def videoDetection():
    global person_model, weapon_model
    text.delete('1.0', END)
    update_status("Selecting video...")
    filename = filedialog.askopenfilename(initialdir="Video", title="Select Video",
                                          filetypes=(("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*")))
    if not filename:
        text.insert(END, "No video file selected. Please select a file.\n\n")
        update_status("No video selected")
        return
    
    text.insert(END, filename + " loaded\n\n")
    update_status("Processing video...")
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        text.insert(END, "Failed to open video. Please try again.\n\n")
        update_status("Error loading video")
        return

    # Get screen dimensions and calculate max dimensions for display
    screen_width = main.winfo_screenwidth()
    screen_height = main.winfo_screenheight()
    max_display_height = screen_height - 200
    max_display_width = screen_width - 200

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    text.insert(END, f"Original Video Resolution: {frame_width}x{frame_height}\n")

    scaling_factor = min(max_display_height / frame_height, max_display_width / frame_width)
    new_width = int(frame_width * scaling_factor)
    new_height = int(frame_height * scaling_factor)
    text.insert(END, f"Resized Video Resolution: {new_width}x{new_height}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (new_width, new_height))  # Resize frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Person detection (YOLOv5s)
        person_count = 0
        if person_model is not None:
            person_results = person_model(frame_rgb)
            person_detections = person_results.xyxy[0].numpy()
            for det in person_detections:
                if int(det[5]) == 0:  # Class 0 is 'Person'
                    xmin, ymin, xmax, ymax = map(int, det[:4])
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
                    cv2.putText(frame, 'Person', (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, GREEN, 2)
                    person_count += 1

        # Weapon detection (YOLOv9)
        weapon_count = 0
        if weapon_model is not None:
            # Preprocess for YOLOv9 (Ultralytics API)
            img = cv2.resize(frame_rgb, (640, 640))  # Resize to model input size
            results = weapon_model(img)  # Ultralytics YOLO inference
            detections = results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, conf, cls]
            
            # Rescale boxes to original frame size
            scale_x = new_width / 640
            scale_y = new_height / 640
            for det in detections:
                if len(det) >= 6 and det[4] >= CONFIDENCE_THRESHOLD:
                    xmin, ymin, xmax, ymax, conf, cls = det[:6]
                    cls = int(cls)
                    if cls < len(weapon_classes):
                        # Rescale coordinates
                        xmin, xmax = map(int, [xmin * scale_x, xmax * scale_x])
                        ymin, ymax = map(int, [ymin * scale_y, ymax * scale_y])
                        label = weapon_classes[cls]
                        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), RED, 2)
                        cv2.putText(frame, label, (int(xmin), int(ymin-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2)
                        weapon_count += 1

        # Display counts
        cv2.putText(frame, f"Person Count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2)
        cv2.putText(frame, f"Weapon Count: {weapon_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
        
        cv2.imshow("Person and Weapon Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    update_status("Video processing complete")

# Initialize main window
main = tk.Tk()
main.title("CCTV AI: Crowd Management & Crime Prevention")
main.geometry("1300x800")  # Adjusted for better fit
main.configure(bg='#f0f4f8')  # Light gray-blue background

# Custom styles
style = ttk.Style()
style.theme_use('clam')  # Modern theme
style.configure('TButton', font=('Helvetica', 12, 'bold'), padding=10, background='#4a90e2', foreground='white')
style.map('TButton', background=[('active', '#357abd')])  # Hover effect
style.configure('TLabel', font=('Helvetica', 14), background='#f0f4f8', foreground='#333')
style.configure('Header.TLabel', font=('Helvetica', 18, 'bold'), foreground='#2c3e50')

# Header frame
header_frame = ttk.Frame(main, padding=10, style='Header.TFrame')
header_frame.pack(fill='x')
style.configure('Header.TFrame', background='#2c3e50')  # Dark header background

# Title label
title = ttk.Label(header_frame, text='CCTV AI: Crowd Management, Crime Prevention & Work Monitoring',
                  style='Header.TLabel', foreground='')
title.pack(pady=10)

# Main content frame
content_frame = ttk.Frame(main, padding=20)
content_frame.pack(fill='both', expand=True)

# Text output frame
text_frame = ttk.LabelFrame(content_frame, text="Output Log", padding=10)
text_frame.pack(fill='both', pady=10, padx=10, expand=True)

# Text widget with scrollbar
text = Text(text_frame, height=15, width=100, font=('Arial', 11), bg='#ffffff', fg='#333', bd=2, relief='sunken')
scroll = ttk.Scrollbar(text_frame, orient='vertical', command=text.yview)
text.configure(yscrollcommand=scroll.set)
scroll.pack(side='right', fill='y')
text.pack(fill='both', expand=True)

# Button frame
button_frame = ttk.Frame(content_frame, padding=10)
button_frame.pack(fill='x', pady=10)

# Buttons with modern styling
loadButton = ttk.Button(button_frame, text="Load YOLOv5 & YOLOv9 Models", command=loadModel)
loadButton.grid(row=0, column=0, padx=10, pady=5)

imageButton = ttk.Button(button_frame, text="Person and Weapon Detection from Images", command=imageDetection)
imageButton.grid(row=0, column=1, padx=10, pady=5)

videoButton = ttk.Button(button_frame, text="Person and Weapon Detection from Videos", command=videoDetection)
videoButton.grid(row=0, column=2, padx=10, pady=5)

graphButton = ttk.Button(button_frame, text="Yolo Training Graph", command=graph)
graphButton.grid(row=1, column=0, padx=10, pady=5)

exitButton = ttk.Button(button_frame, text="Exit", command=main.destroy)
exitButton.grid(row=1, column=1, padx=10, pady=5)

# Status bar
status_var = tk.StringVar(value="Ready")
status_bar = ttk.Label(main, textvariable=status_var, font=('Arial', 10), background='#d9e2ec', foreground='#333',
                       relief='sunken', anchor='w', padding=5)
status_bar.pack(fill='x', side='bottom')

def update_status(message):
    status_var.set(message)
    main.update()

# Center the window on screen
main.update_idletasks()
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
window_width = 1300
window_height = 800
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
main.geometry(f"{window_width}x{window_height}+{x}+{y}")

main.mainloop()