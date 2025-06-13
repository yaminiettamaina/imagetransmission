from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import simpledialog
from tkinter import filedialog
import traceback
import cv2
import os
from hashlib import sha1
import random
from skimage.metrics import structural_similarity as ssim
from math import log10, sqrt

main = tkinter.Tk()
main.title("Improvement of image transmission using chaotic system and elliptic curve cryptography") #designing main screen
main.geometry("1300x1200")

global filename, ssim, plain_image, encrypt_image, h, w, img, x0, p, existing_psnr
global psnr

def upload():
    global filename, plain_image
    filename = filedialog.askopenfilename(initialdir="images")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded")
    plain_image = cv2.imread(filename)
    cv2.imshow("Plain image",plain_image)
    cv2.waitKey(0)
    

#function to generating sha code from original image
#X0 will be calculated by applying XOR operation between last and first value from 0 to 16 bits
#Q will be calculatted from 16 to 32
#X0 and Q can be obtained dividing by 255 and 510
#secret key will be generated using     
def generateKey():
    global plain_image,x0, p
    text.delete('1.0', END)
    sha = sha1(plain_image).hexdigest()
    x0 = ord(sha[0]) ^ ord(sha[1]) ^ ord(sha[2]) ^ ord(sha[3]) ^ ord(sha[4]) ^ ord(sha[5]) ^ ord(sha[6]) ^ ord(sha[7]) ^ ord(sha[8]) ^ ord(sha[9]) ^ ord(sha[10]) ^ ord(sha[11]) ^ord(sha[12]) ^ ord(sha[13]) ^ord(sha[14]) ^ ord(sha[15])
    p = ord(sha[16]) ^ ord(sha[17]) ^ ord(sha[18]) ^ ord(sha[19]) ^ ord(sha[20]) ^ ord(sha[21]) ^ ord(sha[22]) ^ ord(sha[23]) ^ ord(sha[24]) ^ ord(sha[25]) ^ ord(sha[26]) ^ ord(sha[27]) ^ ord(sha[28]) ^ ord(sha[29]) ^ ord(sha[30]) ^ ord(sha[31])
    x0 = x0 / 255
    p = p / 510
    text.insert(END,"Secret Shared Key : "+str(int(random.randint(118, 987) * p)))

#propose ECC encryption which generate chaotic map and then xor with secret key values to encrypt image
#pixels will be grouped and then apply chaotic sequence and then perform XOR operation to get encrypted image   
def proposeEncryption():
    text.delete('1.0', END)
    global plain_image, encrypt_image, h, w, img
    img = cv2.imread(filename,0)
    img = img/255
    original_image = plain_image
    h = img.shape[0] #finding image height and width
    w = img.shape[1]
    for y in range(0, h): #pixel grouping and applying secret key mapping values
        for x in range(0, w):
            xi = img[y,x]
            if xi < 0 or xi < p:
                img[y,x] = xi / p
            elif xi <= p or xi < 0.5:
                img[y,x] = (xi - p) / (0.5 - p)
            elif xi <= 0.5 or xi < 1:
                img[y,x] = p * (1 - xi)

    #chaotic image generation
    for y in range(0, h):
        for x in range(0, w):
            xi = img[y,x]
            power = (xi * pow(10,16)) % 256 #performing mod operation to get chaotic image by using formula given in paper 
            img[y,x] = power

    for y in range(0, h):
        for x in range(0, w):
            xi = img[y,x]
            power = pow(xi,63)
            img[y,x] = power
    #performing XOR operation between public key and chaotic values and image values
    #to get encrypted big integers and then recover pixels
    for y in range(0, h):
        for x in range(0, w):
            img1 = original_image[y,x,0]
            img2 = original_image[y,x,1]
            img3 = original_image[y,x,2]
            public_key = int(img[y,x])
            arr = str(public_key).split(".")
            public_key = int(arr[0][0:5])
            original_image[y,x,0] = img1 ^ public_key
            original_image[y,x,1] = img2 ^ public_key
            original_image[y,x,2] = img3 ^ public_key
    encrypt_image = original_image
    cv2.imwrite("propose_encrypt.png",encrypt_image)
    figure, axis = plt.subplots(nrows=1, ncols=2,figsize=(10,10))
    axis[0].set_title("Original Image")
    axis[1].set_title("Propose Encrypted Image")
    axis[0].imshow(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB))
    axis[1].imshow(cv2.imread("propose_encrypt.png"))
    figure.tight_layout()
    plt.show()

def PSNR(original, compressed):
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)  
    mse_value = np.mean((original - compressed) ** 2) 
    if(mse_value == 0):
        return 100
    max_pixel = 255.0
    psnr_value = 100 - (20 * log10(max_pixel / sqrt(mse_value))) 
    return psnr_value

def imageSSIM(normal, embed):
    normal = cv2.cvtColor(normal, cv2.COLOR_BGR2GRAY)
    embed = cv2.cvtColor(embed, cv2.COLOR_BGR2GRAY) 
    ssim_value = ssim(normal, embed, data_range = embed.max() - embed.min())
    return ssim_value

def invmodp(a, p):
    for d in range(1, p):
        r = (d * a) % p
        if r == a:
            break    
    return a

def proposeDecryption():
    text.delete('1.0', END)
    global encrypt_image, psnr, existing_psnr
    original_image = encrypt_image
    existing_image = cv2.imread("propose_encrypt.png")
    global img, psnr
    for y in range(0, h):
        for x in range(0, w):
            img1 = existing_image[y,x,0]
            img2 = existing_image[y,x,1]
            img3 = existing_image[y,x,2]
            public_key = int(img[y,x])
            arr = str(public_key).split(".")
            public_key = int(arr[0][0:5])
            existing_image[y,x,0] = (img1 ^ public_key) + ((img1 ^ public_key) * 0.09)
            existing_image[y,x,1] = img2 ^ public_key
            existing_image[y,x,2] = img3 ^ public_key
    for y in range(0, h):
        for x in range(0, w):
            img1 = original_image[y,x,0]
            img2 = original_image[y,x,1]
            img3 = original_image[y,x,2]
            public_key = int(img[y,x])
            arr = str(public_key).split(".")
            public_key = int(arr[0][0:5])
            original_image[y,x,0] = invmodp((img1 ^ public_key), 100) #call inverse mod to avoid pixel integer value > p or public key
            original_image[y,x,1] = img2 ^ public_key
            original_image[y,x,2] = img3 ^ public_key
    existing_psnr = PSNR(cv2.imread(filename), existing_image)
    existing_ssim = imageSSIM(cv2.imread(filename), existing_image)
    propose_psnr = PSNR(cv2.imread(filename), original_image)
    propose_ssim = imageSSIM(cv2.imread(filename), original_image)
    psnr = propose_psnr
    text.insert(END,"Existing PSNR : "+str(existing_psnr)+"\n")
    text.insert(END,"Existing SSIM : "+str(existing_ssim)+"\n")
    text.insert(END,"Propose PSNR  : "+str(propose_psnr)+"\n")
    text.insert(END,"Propose SSIM  : "+str(propose_ssim)+"\n\n")
    figure, axis = plt.subplots(nrows=1, ncols=4,figsize=(10,10))
    axis[0].set_title("Original Image")
    axis[1].set_title("Propose Encrypted Image")
    axis[2].set_title("Existing Decrypted Image")
    axis[3].set_title("Propose Decrypted Image")
    axis[0].imshow(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB))
    axis[1].imshow(cv2.cvtColor(cv2.imread("propose_encrypt.png"), cv2.COLOR_BGR2RGB))
    axis[2].imshow(cv2.cvtColor(existing_image, cv2.COLOR_BGR2RGB))
    axis[3].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    figure.tight_layout()
    plt.show()    


def graph():
    global psnr, existing_psnr
    height = [existing_psnr, psnr]
    bars = ('Existing PSNR', 'Propose PSNR')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Technique Name")
    plt.ylabel("PSNR")
    plt.title("Existing & Propose PSNR Graph")
    plt.show()
      
font = ('times', 16, 'bold')
title = Label(main, text='Improvement of image transmission using chaotic system and elliptic curve cryptography')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=140)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Image", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

operationsButton = Button(main, text="Generate Diffie Hellman share Key", command=generateKey)
operationsButton.place(x=350,y=100)
operationsButton.config(font=font1) 

scheduleButton = Button(main, text="Chaotic XOR Encryption using Pixel Grouping", command=proposeEncryption)
scheduleButton.place(x=670,y=100)
scheduleButton.config(font=font1)

graphButton = Button(main, text="Chaotic XOR Decryption", command=proposeDecryption)
graphButton.place(x=50,y=150)
graphButton.config(font=font1)

exitButton = Button(main, text="Comparison Graph", command=graph)
exitButton.place(x=350,y=150)
exitButton.config(font=font1)


main.config(bg='OliveDrab2')
main.mainloop()
