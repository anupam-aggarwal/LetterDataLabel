
# coding: utf-8

# # Importing Libraries

# In[1]:


import cv2
import math
import numpy as np
import matplotlib
import matplotlib.image as mpimg
from PIL import Image as im
from scipy.ndimage import interpolation as inter
from scipy import stats
import os
import shutil
#import matplotlib.pyplot as plt
import sys


# # Code for Segementation to word Level

# In[2]:


# Function to remove background texture --- Will work if texture is comparatively light as compared to text
def bg_filter(image):
    color_select = np.copy(image)

    # defining color criteria
    val = 250
    red_threshold = val
    green_threshold = val
    blue_threshold = val
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]

    # identify pixels above threshold
    thresholds = (image[:, :, 0] > rgb_threshold[0]) | (image[:, :, 1] > rgb_threshold[1]) | (image[:, :, 2] > rgb_threshold[2])
    color_select[thresholds] = [255, 255, 255]

    return cv2.cvtColor(np.array(color_select), cv2.COLOR_RGB2BGR)

def pad_image(img,val):
    if(len(img.shape)==2):
        return cv2.copyMakeBorder(img,2,2,2,2,cv2.BORDER_CONSTANT,value=val)
    elif(len(img.shape)==3):
        return cv2.copyMakeBorder(img,2,2,2,2,cv2.BORDER_REPLICATE)


# Function to create binary image
def img2binary(image):
    color_select = np.copy(image)
    (thresh, im_bw) = cv2.threshold(color_select, 128, 255, cv2.THRESH_BINARY_INV)

    return cv2.cvtColor(np.array(im_bw), cv2.COLOR_RGB2BGR)

# Function to fix skew angle if any?
def skew_fix(image):
    # convert to binary
    image = im.fromarray(image)
    wd, ht = image.size
    pix = np.array(image.convert('1').getdata(), np.uint8)
    bin_img = 1 - (pix.reshape((ht, wd)) / 255.0)

    def find_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        hist = np.sum(data, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        return hist, score

    delta = 0.5
    limit = 7
    angles = np.arange(-limit, limit + delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(bin_img, angle)
        scores.append(score)

    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    print("Best angle for skew correction:", best_angle)
    print()
    # correct skew
    data = inter.rotate(bin_img, best_angle, reshape=False, order=0)
    img = im.fromarray((255 * data).astype("uint8")).convert("RGB")
#         img.save('skew_corrected2.png')
#         plt.imshow(img)
#         plt.show()
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Function to seperate lines and words 
def continuousSubsequence(x,th,diff):
    up = []
    down =[]
    i = 0
    while(i<len(x)-1):
        if(x[i] > th):
            up.append(i)
#             print("up: " +str(i),end='\t')
            i = i+1
            while(not(x[i] <= th) and i<len(x)-1):
                i = i+1
            down.append(i)
#             print("down: " +str(i))
            i = i+1
        else:
            i = i+1
    u = []
    d = []
    for i in range(0,len(up)):
        if(down[i]-up[i]>diff):
            u.append(up[i])
            d.append(down[i])
    return u,d

# Main function to sepearte lines and words from image
def img2line(image):

    # TWEAK RESIZING FACTOR FOR SPACING
    image = cv2.resize(image, (0, 0), fx=1.69, fy=1.69)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H,W = image.shape[0],image.shape[1]
 
    
    th, rotated = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    rotated = cv2.dilate(rotated, None, iterations=3)
    rotated = cv2.erode(rotated, None, iterations=2)
    rotated = pad_image(rotated,0)
    
    hist = cv2.reduce(rotated, 1, cv2.REDUCE_AVG).reshape(-1)
#     plt.plot(hist)
#     print(hist)
    mode = stats.mode(hist)[0][0] 
#     print(mode)
    upper,lower = continuousSubsequence(hist,mode,10)

#     print("uppers:", upper)
#     print("lowers:", lower)

    diff = []
    for k in range(0, len(upper)):
        diff.append(lower[k] - upper[k])
    
    def nearestInt(x):
        f,i = math.modf(x)
        if(f<.6):
            return i
        else:
            return i+1
    print("diff:", diff)
    minim = min(diff)
    for i in range(0, len(diff)):
        diff[i] = int(nearestInt(diff[i] / minim))
        
    print("diff normalised:", diff, "\n")


    def breakImg(up,low,n,points):
        if(n==1):
            return points
        else:
            points = points + [int(((n-1)*up + low)/n)]
            return breakImg(int(((n-1)*up + low)/n),low,n-1,points)

    up = []
    low = []
    for i in range(0,len(diff)):
        if(diff[i] > 1):
            points = breakImg(upper[i],lower[i],diff[i],[])
            up = up + [upper[i]]
            for j in points:
                up = up+[j]
                low = low + [j];
            low = low + [lower[i]]
        else:
            up.append(upper[i])
            low.append(lower[i])

    print("up:", up)
    print("low:", low)

    
    rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
    for y in up:
        cv2.line(rotated, (0, y), (rotated.shape[1], y), (255, 0, 0), 1)

    for y in low:
        cv2.line(rotated, (0, y), (rotated.shape[1], y), (0, 255, 0), 1)

    #cv2.imshow("result.png", rotated)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    def line2words(image,up,down,H,W):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # smooth the image to avoid noises
        gray = cv2.medianBlur(gray, 5)

        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)
        thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        # apply some dilation and erosion to join the gaps
        thresh = cv2.dilate(thresh, None, iterations=4)
        thresh = cv2.erode(thresh, None, iterations=3)
        thresh = pad_image(thresh,0)
        
        hist = cv2.reduce(thresh, 0, cv2.REDUCE_AVG).reshape(-1)
#         plt.plot(hist)
#         print(hist)
        #TODO -- Need to tweak threshhold parameter
        lefts,rights = continuousSubsequence(hist,10,15)

        margin = 3
        ls = []
        for i in range(0,len(lefts)):
            temp = (max(up-margin,0),min(down+margin,H-1),max(lefts[i]-margin-3,0),min(rights[i]+margin,W-1))
            ls.append(temp)
#             print(temp)
        
        return ls
    
    word_list = []
    for i in range(0, len(up)):
        sample_image =cv2.cvtColor(np.array(image[up[i]:low[i],:]), cv2.COLOR_RGB2BGR)
        word_list = word_list+line2words(sample_image,up[i],low[i],H,W)
        
    return word_list


# # Driver code for getting words

# In[3]:

print("Initial")
#print(sys.argv[1])
img = cv2.imread("./page/image.jpg")
# img = cv2.resize(img,(2,2))
ori = np.copy(img)
img = bg_filter(img)
img = img2binary(img)
img = skew_fix(img)
result = img2line(img)
# cv2.imshow("res",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[4]:


b = (np.average(ori[0,:,0]) + np.average(ori[:,0,0]))/2
g = (np.average(ori[0,:,1]) + np.average(ori[:,0,1]))/2
r = (np.average(ori[0,:,2]) + np.average(ori[:,0,2]))/2
back = [int(b),int(g),int(r)]


ori = cv2.resize(ori,(0,0),fx = 1.69,fy = 1.69)
img = cv2.resize(img,(0,0),fx = 1.69,fy = 1.69)
words = [(img[point[0]:point[1],point[2]:point[3]],np.copy(ori[point[0]:point[1],point[2]:point[3]])) for point in result]
words = [(pad_image(img[0],255),pad_image(img[1],0))for img in words]
words = np.asarray(words)

# # Letter Level Segmentation
# #### Author: Anupam Aggarwal

# In[8]:


fSize = [0,0]


# In[9]:


def binaryImages(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    threshed_color = cv2.cvtColor(threshed,cv2.COLOR_GRAY2BGR)
    return gray,threshed,threshed_color

# Function to remove header line -- May Fail if intesity of points is max at points other than at header line
def rmHeaderLine(threshed):
    intensity = cv2.reduce(threshed, 1, cv2.REDUCE_AVG).reshape(-1)    
#     plt.plot(intensity)
#     print(intensity)
    maxRows = [i for i in range(0,len(intensity)) if intensity[i] == np.max(intensity)]
    pixels = 13
    med = maxRows[len(maxRows)//2] if (len(maxRows)%2 == 1) else (maxRows[len(maxRows)//2] + maxRows[len(maxRows)//2-1])/2
    (a,b) = (med - (pixels-1)/2,med + (pixels-1)/2)
    
#     print(max(math.floor(a),0),min(math.ceil(b),len(intensity)))
    # removing header line
    for i in range(max(math.floor(a),0),min(math.ceil(b),len(intensity))):
        threshed[i] = np.array(0)
    
    return (a,b),threshed
            

def verticalSeperation(threshed,th):
    #threshed = pad_image(threshed)
    threshed = cv2.rotate(threshed,cv2.ROTATE_90_CLOCKWISE)

    hist = cv2.reduce(threshed, 1, cv2.REDUCE_AVG).reshape(-1)
#     print(hist)
#     plt.plot(hist)
    th = 2
    upper,lower = continuousSubsequence(hist,th,5)
    
    return upper,lower

def resize2_32(img):
    maxInd = 0 if (img.shape[0] > img.shape[1]) else 1
    fac = 32/img.shape[maxInd]
    img = cv2.resize(img,(0,0),fx=fac,fy=fac)
    if(img.shape[maxInd] != 32):
        newSize = (32,img.shape[1]) if maxInd==0 else (img.shape[0],32)
        img = cv2.resize(img,newSize)
    delta_w = 32 - img.shape[1]
    delta_h = 32 - img.shape[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
#     print("Image Shape " + str(img.shape))
#     print("Margins "+str((top,bottom,left,right)))
    
    if(len(img.shape)==3):
        
        if(top!=0 or bottom != 0):
            b = np.expand_dims(np.ones((top,32),dtype='uint8')*back[0],axis=2)
            g = np.expand_dims(np.ones((top,32),dtype='uint8')*back[1],axis=2)
            r = np.expand_dims(np.ones((top,32),dtype='uint8')*back[2],axis=2)
            im = np.concatenate((b,g),axis=2)
            im = np.concatenate((im,r),axis=2)
#             print(im.shape)

            img = np.concatenate((im,img),axis=0)

            b = np.expand_dims(np.ones((bottom,32),dtype='uint8')*back[0],axis=2)
            g = np.expand_dims(np.ones((bottom,32),dtype='uint8')*back[1],axis=2)
            r = np.expand_dims(np.ones((bottom,32),dtype='uint8')*back[2],axis=2)
            im = np.concatenate((b,g),axis=2)
            im = np.concatenate((im,r),axis=2)
#             print(im.shape)
            img = np.concatenate((img,im),axis=0)
#             print(img.shape)
        
        elif(left!=0 or right != 0):
            b = np.expand_dims(np.ones((32,left),dtype='uint8')*back[0],axis=2)
            g = np.expand_dims(np.ones((32,left),dtype='uint8')*back[1],axis=2)
            r = np.expand_dims(np.ones((32,left),dtype='uint8')*back[2],axis=2)
            im = np.concatenate((b,g),axis=2)
            im = np.concatenate((im,r),axis=2)
#             print(im.shape)
            
            img = np.concatenate((im,img),axis=1)

            b = np.expand_dims(np.ones((32,right),dtype='uint8')*back[0],axis=2)
            g = np.expand_dims(np.ones((32,right),dtype='uint8')*back[1],axis=2)
            r = np.expand_dims(np.ones((32,right),dtype='uint8')*back[2],axis=2)
            im = np.concatenate((b,g),axis=2)
            im = np.concatenate((im,r),axis=2)
#             print(im.shape)
            img = np.concatenate((img,im),axis=1)
# #             print(img.shape)
#         print("ORI")
#         print(img.shape)
#         print()
        return img
    elif(len(img.shape)==2):
        tmp = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,0)
#         print("BW")
        return tmp

def makeCollage(ls):
    length = 0
    if(type(ls)=='np.ndarray'):
        length = ls.shape[0]
    else:
        length = len(ls)
    
    col = math.floor(math.sqrt(length))
    row = length//col
    #print (row,col)
    
    res = ls[0]
    for i in range(1,col):
        res = np.concatenate((res,ls[i]),axis=1)
      
    for i in range(1,row):
        temp = ls[i*col]
        for j in range(1,col):
#             print(temp.shape,ls[i*col+j].shape)
            temp = np.concatenate((temp,ls[i*col+j]),axis=1)
        res = np.concatenate((res,temp))
    
#     cv2.imshow("res",res)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
    rem = length - row*col
    if(rem>0):
        temp = ls[row*col]
        for i in range(1,rem):
            temp = np.concatenate((temp,ls[row*col+i]),axis=1)
        
        for j in range(rem,col):
            tp = np.zeros(ls[1].shape,dtype="uint8")
            temp = np.concatenate((temp,tp),axis=1)
        
        res = np.concatenate((res,temp))
    
    return res

def determineFsize(img):
    global fSize
    kernel = np.ones((3,3),np.uint8)
    img = cv2.dilate(img,kernel,iterations=2)
    img = cv2.blur(img,(3,3))
    img = cv2.erode(img,kernel,iterations=2)
    _,contours,_ = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if (h > 20):
            fSize = [fSize[0]+1,(fSize[0]*fSize[1] + h)/(fSize[0]+1)]
#             cv2.imshow("res",cv2.rectangle(cv2.cvtColor(img,cv2.COLOR_GRAY2BGR),(x,y),(x+w,y+h),(0,255,0),1))
#             cv2.waitKey(500)
#             cv2.destroyAllWindows()

def firstlevelSegment(img):
    img = cv2.resize(img,(0,0),fx = 1.69, fy = 1.69)
    gray, threshed, threshed_color = binaryImages(img)

    # removing header line
    (u,l),threshed = rmHeaderLine(threshed)
    
    # vertically seperating
    lefts,rights = verticalSeperation(threshed,2)
    
    for i in range(0,len(lefts)):
        determineFsize(threshed[int(l):img.shape[0],lefts[i]:rights[i]])
       
    return (u,l),lefts,rights

def thirdLevelSegmentation(img,head,lt,rt):
    img0 = cv2.resize(img[0],(0,0),fx=1.69,fy=1.69)
    img1 = cv2.resize(img[1],(0,0),fx=1.69,fy=1.69)

    _, threshed, threshed_color = binaryImages(img0)

    lst = []
    lst_ori = []
    for i in range(0,len(lt)):
        # lis = []
        letter = threshed[:,lt[i]:rt[i]]
        #letter = threshed
        up = letter[:int(head[0])]
        middle = letter[int(head[1]):int(head[1]+fSize[1]-4)]
        below = letter[int(head[1]+fSize[1]-7):]
        
        letter_ori = img1[:,lt[i]:rt[i]]
        up_ori = letter_ori[:int(head[0])]
        middle_ori = letter_ori[int(head[1]):int(head[1]+fSize[1]-4)]
        below_ori = letter_ori[int(head[1]+fSize[1]-7):]

        if((np.sum(up)/255) > 25):
            up = resize2_32(up)
            up[:3,:3] = 255*np.ones((3,3))
            lst.append(up)
            up_ori = resize2_32(up_ori)
            lst_ori.append(up_ori)
#             cv2.imshow("up_bw",up)
#             cv2.imshow("up_ori",up_ori)

        middle = pad_image(middle,0)
        middle_ori = pad_image(middle_ori,0)
#         print("beforeResizing")
#         print(middle.shape,middle_ori.shape)
        left,right = verticalSeperation(middle,20)
        for n in range(0,len(left)):
            lst.append(resize2_32(middle[:,left[n]:right[n]]))
            lst_ori.append(resize2_32(middle_ori[:,left[n]:right[n]]))
#             cv2.imshow("middle_bw"+str(n),resize2_32(middle[:,left[n]:right[n]]))
#             cv2.imshow("middle_ori"+str(n),resize2_32(middle_ori[:,left[n]:right[n]]))
            
        if((np.sum(below)/255) > 125):
            below = resize2_32(below)
            below[-3:,:3] = 255*np.ones((3,3))
            lst.append(below)
            below_ori = resize2_32(below_ori)
            lst_ori.append(below_ori)
#             cv2.imshow("below_bw",below)
#             cv2.imshow("below_ori",below_ori)
            
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return (lst,lst_ori)


def secondLevelSegmentation(img,up,dn):
    img = cv2.resize(img,(0,0),fx = 1.69, fy = 1.69)
    img = img[int(up):int(dn)]

    gray, threshed, threshed_color = binaryImages(img)
    lefts,rights = verticalSeperation(threshed,10)
    
    return lefts,rights


# In[10]:


def letterSegmentation(ls):
    leftsLs = []
    rightsLs = []
    headerLs = []
    
    # First Level Segmentation
    # getting coordinates for each letter, NECESSARY
    for i in range(0,len(ls)):
        img = ls[i][0]
        head, left, right = firstlevelSegment(img)
        leftsLs.append(left)
        rightsLs.append(right)
        headerLs.append(head)
    print(fSize)
    # Third Level Segmentation
    letters = []
    for i in range(0,len(ls)):
        img = ls[i];
        head = headerLs[i]
        lt = leftsLs[i]
        rt = rightsLs[i]
        letters.append(thirdLevelSegmentation(img,head,lt,rt))

#      Second Level Segmentation
    del leftsLs,rightsLs
    leftsLs = []
    rightsLs = []
    for i in range(0,len(ls)):
        img = ls[i][0]
        up = headerLs[i][1]
        dn = up + fSize[1]-4
        lt,rt = secondLevelSegmentation(img,up,dn)
        leftsLs.append(lt)
        rightsLs.append(rt)
    
    return leftsLs,rightsLs,headerLs,np.asarray(letters)


# In[11]:


leftsLs,rightsLs,headerLs,ls = letterSegmentation(words)



# In[ ]:

letters = []
for word in ls:
    for lt in word[1]:
        letters.append(lt)
res = makeCollage(letters)
cv2.imshow("res",res)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:

'''
BW = []
for word in ls:
    temp = [lt for lt in word[0]]
    BW.append(temp)


# In[ ]:


COL = []
for word in ls:
    temp = [lt for lt in word[0]]
    COL.append(temp)

'''
# In[ ]:

BW = []
for word in ls:
    for lt in word[0]:
        BW.append(lt)

COL = []
for word in ls:
    for lt in word[1]:
        COL.append(lt)


np.save("BW",BW)
np.save("COL",COL)


# In[ ]:


'''
Final result which is to be predicted is Letters.
Ist dimension is Number of words in input paragraph
2nd dimension are Black and white images of letters or modifiers to predicted
'''

