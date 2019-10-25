# Sketch手绘风格处理

Sketch 分别对rgb三通道使用x和y方向的sobel，计算量较大，速度慢；

SketchFast 直接将图像处理成灰度图，再提取边缘，再gray2rgb，速度较快，可以达到实时处理效果。
