import cv2
import glob
import random
import numpy as np

#emotions=["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
emotions=["neutral", "anger", "disgust", "happy", "surprise"]
fishface=cv2.createFisherFaceRecognizer()

faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

data={}

def get_image(camera):
 # read is the easiest way to get a full image out of a VideoCapture object.
 retval, im = camera.read()
 return im

def take_pic():
    camera_port = 0
    ramp_frames = 30
    camera = cv2.VideoCapture(camera_port)
    for i in xrange(ramp_frames):
        temp = get_image(camera)
    print("Taking image...")
    camera_capture = get_image(camera)
    savloc = "C:\\Users\\Aditya\\Desktop\\IIIT-H\\LightMeUp\\camera\\prev.png"
    cv2.imwrite(savloc, camera_capture)
    del(camera)

def process_pic():
    imgs=glob.glob("C:\\Users\\Aditya\\Desktop\\IIIT-H\\LightMeUp\\camera\\prev.png")
    frame=cv2.imread(imgs[0])
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

    #Go over detected faces, stop at first detected face, return empty if no face.
    if len(face) == 1:
        facefeatures = face
    elif len(face_two) == 1:
        facefeatures = face_two
    elif len(face_three) == 1:
        facefeatures = face_three
    elif len(face_four) == 1:
        facefeatures = face_four
    else:
        facefeatures = ""
        
        #Cut and save face
    for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
        print "face found in file: %s" %frame
        gray = gray[y:y+h, x:x+w] #Cut the frame to size
            
        try:
            out = cv2.resize(gray, (200, 200)) #Resize face so all images have same size
            cv2.imwrite("C:\\Users\\Aditya\\Desktop\\IIIT-H\\LightMeUp\\camera\\procprev.png", out) #Write image
        except:
            print "Feil lel"
            pass #If error, pass file

def get_files(emotion):
    files=glob.glob("dataset\\%s\\*" %emotion)
    random.shuffle(files)
    training=files[:int(len(files)*0.8)]
    prediction=files[-int(len(files)*0.2):]
    return training, prediction

def make_sets():
    training_data=[]
    training_labels=[]
    prediction_data=[]
    prediction_labels=[]
    for emotion in emotions:
        training, prediction = get_files(emotion)
        for item in training:
            image=cv2.imread(item)
            gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            training_data.append(gray)
            training_labels.append(emotions.index(emotion))

        for item in prediction:
            image=cv2.imread(item)
            gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))
            
    return training_data, training_labels, prediction_data, prediction_labels

def run_recognizer():
    training_data, training_labels, prediction_data, prediction_labels=make_sets()
    print "Training emotion classifier"
    print "Size of training dataset is:", len(training_labels), "images"
    fishface.train(training_data, np.asarray(training_labels))
    
    print "Predicting unlabeled dataset"
    cnt=0
    correct=0
    incorrect=0
    '''
    for image in prediction_data:
        pred, conf=fishface.predict(image)
        if pred==prediction_labels[cnt]:    #emotions[pred] gives corresp emotion
            correct+=1
            cnt+=1
        else:
            incorrect+=1
            cnt+=1
    #return ((100*correct)/(correct + incorrect))
    '''
def dowork():
    take_pic()
    process_pic()
    
    image=cv2.imread("C:\\Users\\Aditya\\Desktop\\IIIT-H\\LightMeUp\\camera\\procprev.png")
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pred, conf=fishface.predict(gray)
    return emotions[pred]

metascore=[]
for i in range(1):
    correct=run_recognizer()
correct=dowork()
print correct
