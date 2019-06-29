import cv2 as cv
import numpy as np
import random
import sys

if __name__ == '__main__':
    img_path = sys.argv[1]

    # 2a: read and display the image
    img = cv.imread(img_path)
    b=img[:,:,0]
    g=img[:,:,1]
    r=img[:,:,2]
    cv.imshow('original image', img)


    # 2b: display the intenstity image
    I = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('intensity image',I)
 	
    # 2c: for loop to perform the operation
    halfI = I * 0.5
    diffBlue=np.empty_like(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            diffBlue[y,x] = img[y,x] - b[y,x]
    diffBlue[diffBlue < 0]= max(b[y,x],halfI[y,x],0)


    diffGreen=np.empty_like(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            diffGreen[y,x] = img[y,x] - g[y,x]
    diffGreen[diffGreen < 0]= max(g[y,x],halfI[y,x],0)


    diffRed=np.empty_like(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            diffRed[y,x] = img[y,x] - r[y,x]
    diffRed[diffRed < 0]= max(r[y,x],halfI[y,x],0)
    cv.imshow('Difference image original-blue using for loops',diffBlue)
    cv.imshow('Difference image original-green using for loops',diffGreen)
    cv.imshow('Difference image original-red using for loops',diffRed)                        


    # 2d: one-line statement to perfom the operation above
    cv.imshow('Difference image original-blue using one line',img - np.expand_dims(b,axis=2))
    cv.imshow('Difference image original-green using one line',img - np.expand_dims(g,axis=2))
    cv.imshow('Difference image original-red using one line',img - np.expand_dims(r,axis=2))

    # 2e: Extract a random patch
    patch_center=np.array([150,240])
    patch_y=patch_center[0] - 8
    patch_x=patch_center[1] -8
    patch_image=img[patch_y:patch_y+16,patch_x:patch_x+16]
    print(patch_image.shape)
    cv.imshow('patch_image',patch_image)
    from random import randint
    randposY=randint(16,284)
    randposX=randint(16,464)
    img[randposY:randposY+16,randposX:randposX+16]=patch_image.copy()
    cv.imshow('Modified Image',img)


    # 2f: Draw random rectangles and ellipses
    for x in range(10):
        x1 = randint(0, img.shape[0])
        y1 = randint(0, img.shape[1])
        c1 = randint(0, img.shape[0])
        r1 = randint(0, img.shape[1])
        R=randint(0,255)
        G=randint(0,255)
        B=randint(0,255)
        rectangleImages=cv.rectangle(img,(y1,x1),(r1,c1),(B,G,R),-1)
        cv.imshow('Rectangle image',rectangleImages)

    # draw ellipses
    for x in range(10):
        randRotation = randint(0,90)
        y1 = randint(50, img.shape[0]-50)
        x1 = randint(100, img.shape[1]-100)
        ellipseImages=cv.ellipse(img,(x1,y1),(100,50),randRotation,0,360,255,-1)
        cv.imshow('Ellipse images',ellipseImages)



    # destroy all windows
    cv.waitKey(0)
    cv.destroyAllWindows()  















