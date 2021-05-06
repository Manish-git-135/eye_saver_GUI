import cv2
from tkinter import *
from PIL import ImageTk, Image
from imutils.video import VideoStream
import cv2
import brightness
import distance
import eye

root = Tk()
root.geometry('850x550')
root.resizable(width=0, height=0)
root.title("Eye Saver")
root.configure(bg='white')

# Create a frame
app = Frame(root, bg="white")
app.place(x=505, y=23)
lmain = Label(app)  # Create a label in the frame
lmain.grid()

# Capture from camera
cap = cv2.VideoCapture(0)

# function for video streaming


def video_stream():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(1, video_stream)


video_stream()
# images for buttons
distance_btn_image = PhotoImage(file='./images/distance.png')
eyeBlinkCount_btn_image = PhotoImage(file='./images/eyeCount.png')
start_btn_image = PhotoImage(file="./images/start.png")

# function for find face area
def get_area():
    dist = distance.get_face_area(15000, cap)
    dist_label['text'] = dist
    if dist <= 15000:
        dist_label['bg'] = "#008000"
    elif dist >= 15000:
        dist_label['bg'] = '#FF0000'


# Button for face area
Button(root, image=distance_btn_image, border=0, borderwidth=0, highlightthickness=0,
       fg='white', command=get_area).place(x=110, y=110)

# function for count blink


def get_eye_blink_count():
    count = eye.eye_blink_counter(cap)
    count_label['text'] = count


# button for count eye blink count
Button(root, image=eyeBlinkCount_btn_image, border=0, borderwidth=0, highlightthickness=0,
       fg='white', command=get_eye_blink_count).place(x=110, y=210)

# button for
Button(root, image=start_btn_image, border=0, borderwidth=0, highlightthickness=0,
       fg='white').place(x=90, y=380)

v1 = DoubleVar()
v1.set(brightness.current_brightness)
Scale(root, from_=1, to=100, bg="white", label="Brightness",
      highlightthickness=0, length=200, command=lambda val: brightness.set_brightness(val), variable=v1).place(x=300, y=100)

Scale(root, from_=1, to=100, bg="white", label="Yellow",
      highlightthickness=0, length=200).place(x=400, y=100)

dist_label = Label(root, text="", pady=12, padx=18,
                   bg='#B9C7C4', fg='white')
dist_label.place(x=420, y=380)
count_label = Label(root, text="", pady=12, padx=18,
                    bg="#B9C7C4")
count_label.place(x=590, y=380)


root.mainloop()
