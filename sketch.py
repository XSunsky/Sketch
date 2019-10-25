# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 16:17:36 2019

@author: 56472
"""

import cv2
import numpy as np
from tqdm import tqdm

def sketch(img, style, THRESHOLD):
    if style == 1:
        height, width, _ = img.shape
        if width > height:
            left_or_up = img[:, :int(width/2), :]
            right_or_down = img[:, int(width/2):, :]
        else:
            left_or_up = img[:int(height/2),:, :]
            right_or_down = img[int(height/2):, :, :]
        nodealImg = left_or_up
        dealImg = right_or_down
    elif style == 2:
        nodealImg = img
        dealImg = img
    
    # 彩色边缘检测
    VG, A, PPG = colorgrad(dealImg)
    
    # 阈值处理
    ppg = PPG.astype(np.uint8)
    ppgf = 255 - ppg
    M, N = ppgf.shape
    outputGray = np.zeros([M, N])
    
    for i in range(M):
        for j in range(N):
            if ppgf[i, j] < THRESHOLD:
                outputGray[i, j] = 0
            else:
                outputGray[i, j] = 235 / (255 - THRESHOLD) * (ppgf[i, j] - THRESHOLD)
    
    outputGray = outputGray.astype(np.uint8)
    outputRGB = gray2rgb(outputGray)
    outputRGB = outputRGB.astype(np.uint8)
    
    if M > N:
        combination = np.hstack((nodealImg, outputRGB))
    else:
        combination = np.vstack((nodealImg, outputRGB))

    outputImg = combination
    return outputImg
    

# 梯度算法计算彩色图像的边缘 
def colorgrad(dealImg, Thres=0):
    # 使用sobel算子，计算三个分量图像的x,y偏导数， 
    RX = cv2.Sobel(dealImg[:,:,0], cv2.CV_64F, 1, 0, ksize=5)
    RY = cv2.Sobel(dealImg[:,:,0], cv2.CV_64F, 0, 1, ksize=5)
    GX = cv2.Sobel(dealImg[:,:,1], cv2.CV_64F, 1, 0, ksize=5)
    GY = cv2.Sobel(dealImg[:,:,1], cv2.CV_64F, 0, 1, ksize=5)
    BX = cv2.Sobel(dealImg[:,:,2], cv2.CV_64F, 1, 0, ksize=5)
    BY = cv2.Sobel(dealImg[:,:,2], cv2.CV_64F, 0, 1, ksize=5)
    
    # compute the parameters of the vector gradient. 
    gxx = np.power(RX,2) + np.power(GX,2) + np.power(BX,2)
    gyy = np.power(RY,2) + np.power(GY,2) + np.power(BY,2)
    gxy = RX*RY + GX*GY + BX*BY
    
    eps = 1e-6 * np.ones_like(gxx)
    
    A = 0.5 * np.arctan(2 * gxy / (gxx - gyy + eps))
    G1 = 0.5 * ((gxx + gyy) + (gxx - gyy) * np.cos(2*A) + 2 * gxy * np.sin(2*A))
    
    A = A + np.pi/2
    G2 = 0.5 * ((gxx + gyy) + (gxx - gyy) * np.cos(2*A) + 2 * gxy * np.sin(2*A))
    
    G1 = np.sign(G1) * np.power(np.abs(G1), 0.5)
    G2 = np.sign(G2) * np.power(np.abs(G2), 0.5)
    
    VG  = np.maximum(G1, G2)
    
    # compute the per-plane gradients 
    RG = np.sqrt(np.power(RX,2), np.power(RY,2))
    GG = np.sqrt(np.power(GX,2), np.power(GY,2))
    BG = np.sqrt(np.power(BX,2), np.power(BY,2))
    
    # form the composite by adding the individual results and scale to [0,1] 
    PPG = RG + GG + BG
    PPG = 255 * PPG / (max(map(max,PPG))+1e-6)
    
    
    if Thres != 0:
        VG = (VG>Thres) * VG
        PPG = (PPG > Thres) * PPG
    
    return VG, A, PPG
    
def gray2rgb(img):
    rows, cols = img.shape
    r = np.zeros([rows, cols])
    g = np.zeros([rows, cols])
    b = np.zeros([rows, cols])
    r = np.double(img)
    g = np.double(img)
    b = np.double(img)
    
    outputRGB = cv2.merge([r,g,b]) 
    
    return outputRGB

def ImgSketch(imgDir, mode, save_or_not):
    img = cv2.imread(imgDir)
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thres, _ = cv2.threshold(imggray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    #img = cv2.cvtColor(img, cv.COLOR_BGR2RGB)
#    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    
    res = sketch(img, mode, thres)
    
    if save_or_not == 1:
        cv2.imwrite('imgSketch.jpg', res)

    cv2.imshow('', res)
    cv2.waitKey()
    cv2.destroyAllWindows()

def VideoSketch(VideoDir, mode, save_or_not):
    cap = cv2.VideoCapture(VideoDir)
    
    counts = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rate = int(cap.get(cv2.CAP_PROP_FPS))
    Frameheight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Framewidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    
    if mode == 1:
        height = Frameheight
        width = Framewidth
    elif mode == 2:
        if Frameheight > Framewidth:
            height = Frameheight
            width = 2 * Framewidth
        else:
            height = 2 * Frameheight
            width = Framewidth
    
    out = cv2.VideoWriter('videoSketch.avi', cv2.VideoWriter_fourcc(*'XVID'), rate, (width, height))

    pbar = tqdm(total=100)

    for i in range(counts):
        ret, frame = cap.read()
        
        if i % rate == 0:
            pbar.update(rate/counts)

        if ret == True:                
            framegray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            thres, _ = cv2.threshold(framegray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            
            #img = cv2.cvtColor(img, cv.COLOR_BGR2RGB)
            #frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            
            res = sketch(frame, mode, thres)
            
            if save_or_not == 1:
                out.write(res)
            elif save_or_not == 0:
                cv2.imshow('', res)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break
    
    pbar.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    Dir = '4.mp4'
    style = int(input('输入1或2，选择风格： '))
    save_or_not = int(input('是否保存？（是按1，否按0）：'))
    
    Img_or_Video = Dir.split('.')[-1]
    Imgstyle = ['jpg', 'png', 'gif']
    Videostyle = ['mp4', 'avi', 'rmvb']
    
    if Img_or_Video in Imgstyle:
        ImgSketch(Dir, style, save_or_not)
    elif Img_or_Video in Videostyle:
        VideoSketch(Dir, style, save_or_not)
        
