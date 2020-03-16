import tkinter
from tkinter import Frame
from tkinter import ttk
from tkinter import filedialog

import cv2

tk = tkinter.Tk()
tk.title("Face Detection")

frame = Frame(tk, width=250, height=250)
frame.pack()

face_cascade = cv2.CascadeClassifier('DATA/haarcascades/haarcascade_frontalface_alt2.xml')

def mainmenu():
    B1 = ttk.Button(tk, text="Image", command=Image)
    B1.place(x=100, y=50)

    B2 = ttk.Button(tk, text="Video", command=Video)
    B2.place(x=100, y=100)


def Image():
    path = filedialog.askopenfilename()

    if len(path) > 0:
        image = cv2.imread(path)
        new_image = detect_face(image)

        while True:
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            cv2.imshow("Image", new_image)
            cv2.resizeWindow("Image", 700, 600)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()


def Video():
    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame1 = cap.read()

        frame1 = detect_face(frame1)

        cv2.imshow("Video", frame1)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_face(image):

    face_img = image.copy()

    face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x+w, y+h), (255, 255, 255), 10)

    return face_img

mainmenu()

tk.mainloop()