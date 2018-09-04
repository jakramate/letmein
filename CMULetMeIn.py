from tkinter import *
from tkinter import ttk
import cv2 
import copy
from PIL import Image, ImageTk 
from os import makedirs

from faceDetector import *

# main window
main = Tk()
main.title('CMU Let Me In')
main.geometry('500x700')
 
# gives weight to the cells in the grid
rows = 0
while rows < 50:
    main.rowconfigure(rows, weight=1)
    main.columnconfigure(rows, weight=1)
    rows += 1
 
# Defines and places the notebook widget
nb = ttk.Notebook(main)
nb.grid(row=1, column=0, columnspan=50, rowspan=49, sticky='NESW')
 
vc = cv2.VideoCapture(0)
fr = CMUFaceRecogniser()

def show_frame(event=None):

    # working out which is the current Tab
    tabId = nb.index(nb.select())
                
    if tabId == 0:
        e2.delete(0,'end') # delete entry's text of tab 2
        
        retrain = False
        try:
            if event.widget['text'] == 'Retrain':
                retrain = True
        except:
            pass

        if retrain:
            fr.retrain()

        _, im = vc.read()
        # colorspace conversion needed 
        # cv2 works in BGR mode while Tk works in RGB 
        faces_roi = fr.detect_face(im)
        im = fr.recognise_face(im, faces_roi)

        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGBA)
        im = Image.fromarray(im)    
        im = ImageTk.PhotoImage(image=im)
        
        panel.configure(image=im)
        panel.image = im
        
        main.after(1, show_frame)

    elif tabId == 1:
        # this is frame2
        folderName = e2.get()

        # display image
        _, im = vc.read()
        faces_roi = fr.detect_face(im)
        roi_color = None
        for (x,y,w,h) in faces_roi:
            roi_color = copy.deepcopy(im[y:y+h, x:x+w])
            cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 2)
        
        capture = False
        try:
            if event.widget['text'] == 'Capture':
                capture = True
        except:
            pass

        # show video content here but without face recogntion
        if len(folderName) > 1 and capture:
            if not path.exists(folderName):
                # creating a folder for new user
                makedirs(folderName)
                info = folderName + '/(0)'
                labels.append(Label(page2, text=info))
                names.append(folderName)
                counts.append(0)

            # save figure to new dir
            if len(roi_color) > 50:
                filename = folderName + '/' + str(countFiles(folderName)) + '.jpg'
                roi_color = cv2.resize(roi_color, (100,100), interpolation = cv2.INTER_CUBIC)
                cv2.imwrite(filename, roi_color)
                print(filename)

            # updating image counts for this user
            counts[names.index(folderName)] += 1
            
            # updating the frame
            info = folderName + '/' + '(' + str(counts[names.index(folderName)]) + ')'
            label = labels[names.index(folderName)]
            label.configure(text=info)
            label.pack()
        
        #im = cv2.resize(im, (300,300), interpolation = cv2.INTER_CUBIC)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGBA)
        im = Image.fromarray(im)    
        im = ImageTk.PhotoImage(image=im)

        panel2.configure(image=im)
        panel2.image = im

        #print('Exucuting frame 2')
        if not capture:
            main.after(1, show_frame)

    elif tabId == 2: 
        # this is frame3        
        #print('Executing frame 3')
        # clear e2 
        e2.delete(0,'end')

    

def countFiles(folderName):
    k = [f for f in listdir(folderName) if path.isfile(path.join(folderName,f))]
#    print(k)
    return(len(k))

# ================================================
# Adds tab 1 of the notebook and the stuffs in it
page1 = ttk.Frame(nb)
nb.add(page1, text='Main')
panel = Label(page1)
panel.pack()
b1 = Button(page1, text='Retrain')
b1.pack()
b1.bind('<Button-1>', show_frame)

# =================================================
# Adds tab 2 of the notebook

# page 2 layout
page2 = ttk.Frame(nb)
nb.add(page2, text='Training')
panel2 = Label(page2)
panel2.pack()

e2 = Entry(page2)
e2.pack()
b2 = Button(page2, text='Capture')
b2.pack()
b2.bind('<Button-1>', show_frame)

Label(page2, text='Current users').pack()
labels = [] 
names  = []
counts = [] 

folders = listdir()
for folder in folders:
    if path.isdir(folder) and '__' not in folder:
        info = folder + '/' + '(' + str(countFiles(folder)) + ')'
        labels.append(Label(page2, text=info ))
        labels[-1].pack()
        counts.append(countFiles(folder))
        names.append(folder)


# =================================================
# Adds tab 3 of the notebook
page3 = ttk.Frame(nb)
nb.add(page3, text='Configure')
nb.bind('<<NotebookTabChanged>>', show_frame)
Button(page3, text='In page 3').pack()

# ================ main loop ======================
#main.after(0, show_frame)
try:
    main.mainloop()
finally:
    if vc.isOpened():
        vc.release()
