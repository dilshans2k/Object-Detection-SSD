# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 15:47:30 2020

@author: Dilshan Sandhu
"""
#ssd_300 file is pretrained ssd model
import torch                        #contains pytorch has dynamic graphs which is used to 
                                    #calculate gradients of compostition functions in backward propagation 
from torch.autograd import Variable #responsible for gradient descent will help find torch variable
                                    #(element of dynamic graph) which contains the both tensor and a gradient
import cv2                          #just to draw the rectangles
from data import BaseTransform, VOC_CLASSES as labelmap
#data is just a folder that contains BaseTransform and VOC_CLASSES classes
#BaseTransform will be used to tranform the image so that it is compatible with our neural network
#VOC_CLASSES is just a dictionary that will do the encoding of classes, to do mapping with numbers(as we dont use text)
from ssd import build_ssd           #ssd is library for single shot detection model
                                    #build_ssd will be constructor of ssd neural network, to build the architecture of ssd model
import imageio                      #to process images of videos and applying detect function on images

#frame by frame
#detect will work on single images
#net is neural network
#transform for right format of images to be used in neural network
def detect(frame, net, transform):
    #frame.shape list return 3 arguments [0]-> height [1]->width [2]->no. of channels.. for b/w 1 channel for colored 3 channels
    height,width = frame.shape[:2] 
# =============================================================================
#     now we will do multiple transformations
#     1. first tranform
#     2. then convert transform frame from a numpy array to torch tensor(advanced matrix)
#     3. add a fake dimension to torch tensor and that fake dimension will correspond to batch
#     4. convert to torch variable
# =============================================================================
    #1transform() returns two elements and first one contains numpy array as a frame with required dimensions/size and colors
    frame_t = transform(frame)[0]
    #2
    x = torch.from_numpy(frame_t)
    #2a currently the frame is red blue green and model was trained on green red blue
    x = x.permute(2,0,1)
    #3 fake dimension... neural network accepts only batches and not single input vector or single input image
    #first dimension corresponds to batch and second corresponds to input
    #4 torch variable
    x = Variable(x.unsqueeze(0))
    #feed torch tensor variable into already trained neural network
    y = net(x)
    #new tensor variable that contains the output variables wheter dog cat
    #here we are getting data of torch Variable, and we know there are two parts of a torch variable
    #tensor and a gradient  and by getting the data part of our torch variable we are accessing the tensor part
    detections = y.data
    #new tensor object that will have the dimensions width height width height 
    #position of detected objects inside the image has to be normalised between 0 and 1 (to do this we need scale tensor with 4 dim.)
    #first two width height corresponds to upperleft and last two corresponds to bottom right
    scale = torch.Tensor([width,height,width,height])
# =============================================================================
#     So like we had batch of inputs we also have batch of outputs
#     detections tensor contains 4 things
#     detections = [batch, no. of classes, no. of occurence of each class, (score, x0, y0, x1, y1)]
#     (1) batch output,
#     (2) no. of objects detected i.e. dog,cat,lion,tiger
#     (3) no of occurence corresponds to how many occurence of each class seen i.e if there are 2 dogs, so it will be 2
#     (4) a tuple of 5 elements.. a score for each occurence of each class and coordinates of upper left and lower right corner
#     score corresponds to if it is less than 0.6 then coordinates will not be given(occurence not founnd) 
#     and if greater than 0.6 then coordinates will be given(occurence found)
# =============================================================================
    #detections.size(1) corresponds to number of classes
    #check if detections[1] is also correct?
    for i in range(detections.size(1)):
        #j corresponds to the occurence of the class
        j=0
        #here we will check if greater than 0.6 then we will take that occurence
        #so occurence j of class i of batch 0 and at 4th 0 corresponds to score
        while detections[0,i,j,0] >=0.6:
            #pt corresponds to coordinate of point taken
            #multiplying with scale will normalise so that we can get the coordinates of these points
            #at the scale of the image
            #and as cv2 works with numpy array and we have torch tensor so converting to numpy
            pt = (detections[0,i,j,1:] * scale).numpy()
            #rectangle(image,(x0,y0),(x1,y1),(colors),thickness)
            cv2.rectangle(frame,(int(pt[0]),int(pt[1])),(int(pt[2]),int(pt[3])),(255,0,0),2)
            #putText(image,text to display(obtained by voc_classes mapping) ith class and -1 for 0indeing, 
            #,where to display the label,font,size, color, thickness, continous line and not dots)
            cv2.putText(frame,labelmap[i-1],(int(pt[0]),int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255),2, cv2.LINE_AA)
            j=j+1
    
    return frame

#Creating or actually building the SSD neural network
net = build_ssd('test') #two options available train or test.. but as we are using a pretrained model
#load the weights from pretrained model
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location=lambda storage, loc: storage)) #torch.load will open a tensor that will contain the weights

   
#Creating the transformation
#so that input frames from video are compatible with neural network
#(net.size -> target size of images that will be fed to neural net)
#triplet that will allow to put the colors values at the rightscale - rightscale is the scale under which neural network was trained
transfrom = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))
 
    
#Object Detection on video
reader = imageio.get_reader('animal.mp4')    
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('output_animal.mp4', fps=fps)
for i,frame in enumerate(reader):
    frame = detect(frame, net.eval(), transfrom)
    writer.append_data(frame)
    print(i)#no of frames processed
writer.close()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    