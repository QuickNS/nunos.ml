from __future__ import print_function
import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2
from imutils import resize

def decode(im):
    # Find barcodes and QR codes
    decodedObjects = pyzbar.decode(im)
    return decodedObjects

def readBarcode(image):
    decodedObjects = decode(image)
    barcodes =[]
    for obj in decodedObjects:
        barcodes.append(obj.data.decode('ascii'))
    return barcodes
    
im = cv2.imread('barcode_01.jpg')

barcodes = np.array([])
for i in [200, 500, 1000]:
    t = resize(im, width=i)
    barcodes= np.unique(np.concatenate((barcodes, readBarcode(t))))

print(barcodes)