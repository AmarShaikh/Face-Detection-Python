from PIL import Image
import cv2


def combineImg():

    img1=Image.open('AS1.jpg')
    # img1.show()
    img2=Image.open('AS2.jpg')
    # img2.show()

    # resize image
    img1=img1.resize((626,440))
    img2=img2.resize((626,440))


    #Get Image size
    img1_size=img1.size
    img2_size=img2.size

    # Create new image
    new_image = Image.new('RGB',(2*img1_size[0], img2_size[1]), (250,250,250))

    new_image.paste(img1,(0,0))
    new_image.paste(img2,(img1_size[0],0))

    new_image.save("Merge.jpg","JPEG")

    new_image.show()


def detectFace():
    trainedData=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # load image
    img=cv2.imread('Merge.jpg')

    # converting color to gray scale image
    grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # get face cordinat
    face_Cordinate=trainedData.detectMultiScale(grayImg)

    
    # Draw square  around face

    for (x,y,w,h) in face_Cordinate:    
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,256,0),2)
    
    # print face face_Cordinate
    print(face_Cordinate)
    # show image
    cv2.imshow('Frame',img)
    cv2.waitKey()

detectFace()
