import cv2
import sys
import create_csv
import pandas as pd
import numpy as np

if len(sys.argv)<2:
	print ("Please add Test Image Path")
	sys.exit()

# test_img = sys.argv[1]
test_img = "test.jpg"

faceCascade = cv2.CascadeClassifier('haarcascade_face.xml')

def train():
	
	# Create 'train_faces.csv', which contains the images and their corresponding labels
	create_csv.create()
	
	# Face Recognizer using Linear Binary Pattern Histogram Algo
	face_recognizer = cv2.face.LBPHFaceRecognizer_create()
	
	# Read csv file using pandas
	data = pd.read_csv('train_faces.csv').values
	
	images=[]
	labels=[]
	
	for ix in range(data.shape[0]):
		
		img = cv2.imread(data[ix][0])
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		images.append(gray)
		labels.append(data[ix][1])
	
	face_recognizer.train(images,np.array(labels))
	return face_recognizer
	
	
def resize_image(image, max_width=400, max_height=400):
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        resized_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return resized_image
    return image


def test(test_img, face_recognizer):
    # Đọc ảnh kiểm tra
    image = cv2.imread(test_img)
    if image is None:
        print(f"Không thể đọc ảnh: {test_img}")
        return

    # Chuyển ảnh sang grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Thay đổi kích thước ảnh nếu cần
    gray = resize_image(gray)
    image = resize_image(image)
    
    # Phát hiện khuôn mặt
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=0
    )
    
    for (x, y, w, h) in faces:
        # Vẽ khung xung quanh khuôn mặt
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Dự đoán nhãn khuôn mặt
        pred_label = face_recognizer.predict(gray)
        
        # Hiển thị nhãn trên ảnh
        cv2.putText(image, str(pred_label), (x, y-5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)
    
    # Hiển thị ảnh kết quả
    cv2.imshow('Face Recognition', image)
    cv2.waitKey(0)

	

if __name__ == '__main__':
	face_recog = train()
	test(test_img, face_recog)