# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:47:15 2019

@author: 56472
"""

import cv2
import numpy as np
from tqdm import tqdm


def sketch(img):
#    Img = np.asarray(Image.open(ImgDir).convert('L')).astype('float')     #取得图像灰度
    
    Img = np.asarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype('float')
    
    # 求梯度，并稍作处理
    depth = 10.                                     # (0-100)
    grad = np.gradient(Img)                           # 取图像灰度的梯度值
    grad_x, grad_y = grad                           # 分别取横纵图像梯度值
    grad_x = grad_x*depth/100.
    grad_y = grad_y*depth/100.
    A = np.sqrt(grad_x**2 + grad_y**2 + 1.)
    uni_x = grad_x/A
    uni_y = grad_y/A
    uni_z = 1./A
    
    el = np.pi/2.2                              # 光源的俯视角度，弧度值
    az = np.pi/4                               # 光源的方位角度，弧度值
    dx = np.cos(el)*np.cos(az)              # 光源对x轴的影响
    dy = np.cos(el)*np.sin(az)              # 光源对y轴的影响
    dz = np.sin(el)                             # 光源对z轴的影响
    
#    dx = 1
#    dy = 1
#    dz = 1
    
    gd = 255*(dx*uni_x + dy*uni_y + dz*uni_z)        # 光源归一化
    gd = gd.clip(0,255)                               #避免数据越界，将生成的灰度值裁剪至0-255之间
    
#    im = Image.fromarray(gd.astype('uint8'))         # 重构图像
    imGray = gd.astype('uint8')
    
#    imRGB = cv2.merge([imGray, imGray, imGray])
    imRGB = cv2.cvtColor(imGray, cv2.COLOR_GRAY2BGR)
    
    return imRGB

def dealStyle(img, style):
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
    
    outputRGB = sketch(dealImg)
    M, N, _ = outputRGB.shape
    
    if M > N:
        combination = np.hstack((nodealImg, outputRGB))
    else:
        combination = np.vstack((nodealImg, outputRGB))

    outputImg = combination
    return outputImg


def ImgSketch(imgDir, style, save_or_not):
    img = cv2.imread(imgDir)

    res = dealStyle(img, style)
    
    if save_or_not == 1:
        cv2.imwrite('imgSketch.jpg', res)

    cv2.imshow('', res)
    cv2.waitKey()
    cv2.destroyAllWindows()

def VideoSketch(VideoDir, style, save_or_not, ratio = 10):
    cap = cv2.VideoCapture(VideoDir)
    
    counts = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rate = int(cap.get(cv2.CAP_PROP_FPS))
    Frameheight = int((cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) / ratio)
    Framewidth = int((cap.get(cv2.CAP_PROP_FRAME_WIDTH)) / ratio)

    
    if style == 1:
        height = Frameheight
        width = Framewidth
    elif style == 2:
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
        frame = cv2.resize(frame, None, fx=(1/ratio), fy=(1/ratio), interpolation=cv2.INTER_AREA)
        
        
        if (i+1) % rate == 0:
            pbar.update(100*rate/counts)

        if ret == True:                            
            res = dealStyle(frame, style)
            
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
    Dir = '2.mp4'
    ratio = 1   # 缩小分辨率比例
    style = int(input('输入1或2，选择风格： '))
    save_or_not = int(input('是否保存？（是按1，否按0）：'))
    
    Img_or_Video = Dir.split('.')[-1]
    Imgstyle = ['jpg', 'png', 'gif']
    Videostyle = ['mp4', 'avi', 'rmvb']
    
    if Img_or_Video in Imgstyle:
        ImgSketch(Dir, style, save_or_not)
    elif Img_or_Video in Videostyle:
        VideoSketch(Dir, style, save_or_not, ratio)