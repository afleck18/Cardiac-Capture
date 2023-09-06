
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import numpy as np
from statistics import mean
import PIL.Image
import PIL.ImageStat

from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename


global file
global count
global pixmm
global random
global image_result
global ct

ct = 0
random = 0
count = 0

### Opens Image File from Directory ###
#----------------------------------------------------------------------#
def openFile():
    global file
    file = askopenfilename()
    root.destroy()
    
### Processes image into bitwise file ###
#----------------------------------------------------------------------#
def preproces_image(
    image,
    *,
    kernel_size=15,
    crop_side=50,
    blocksize=35,
    constant=15,
    max_value=255,
):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bit = cv2.bitwise_not(gray)
    image_adapted = cv2.adaptiveThreshold(
        src=bit,
        maxValue=max_value,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=blocksize,
        C=constant,
    )
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    erosion = cv2.erode(image_adapted, kernel, iterations=2)
    return erosion[crop_side:-crop_side, crop_side:-crop_side]

### Finds edges of image allowing for proper cropping ###
#----------------------------------------------------------------------#

def find_edges(image_preprocessed, *, bw_threshold=150, limits=(0.2, 0.15)):
    mask = image_preprocessed < bw_threshold
    edges = []
    for axis in (1, 0):
        count = mask.sum(axis=axis)
        limit = limits[axis] * image_preprocessed.shape[axis]
        index_ = np.where(count >= limit)
        _min, _max = index_[0][0], index_[0][-1]
        edges.append((_min, _max))
    return edges

### Adjusts edges for proper indexing ###
#----------------------------------------------------------------------#

def adapt_edges(edges, *, height, width):
    (x_min, x_max), (y_min, y_max) = edges
    x_min2 = x_min
    x_max2 = x_max + min(250, (height - x_max) * 10 // 11)
    # could do with less magic numbers
    y_min2 = max(0, y_min)
    y_max2 = y_max + min(250, (width - y_max) * 10 // 11)
    return (x_min2, x_max2), (y_min2, y_max2)

### Crops wasted space from image borders ###
#----------------------------------------------------------------------#

def cropImg(img_title):

    if __name__ == "__main__":
        image = cv2.imread(img_title)
        height, width = image.shape[0:2]
        image_preprocessed = preproces_image(image)
        edges = find_edges(image_preprocessed)
        (x_min, x_max), (y_min, y_max) = adapt_edges(
            edges, height=height, width=width
            )
        image_cropped = image[x_min:x_max, y_min:y_max]
        
        return image_cropped

### Detects color within images allowing for proper filtering ###
#----------------------------------------------------------------------#

def detect_color_image(img_title, thumb_size=150, MSE_cutoff=22, adjust_color_bias=True):
    pil_img = PIL.Image.open(img_title)
    bands = pil_img.getbands()
    if bands == ('R','G','B') or bands== ('R','G','B','A'):
        thumb = pil_img.resize((thumb_size,thumb_size))
        SSE, bias = 0, [0,0,0]
        if adjust_color_bias:
            bias = PIL.ImageStat.Stat(thumb).mean[:3]
            bias = [b - sum(bias)/3 for b in bias ]
        for pixel in thumb.getdata():
            mu = sum(pixel)/3
            SSE += sum((pixel[i] - mu - bias[i])*(pixel[i] - mu - bias[i]) for i in [0,1,2])
        MSE = float(SSE)/(thumb_size*thumb_size)
        if MSE <= MSE_cutoff:
            
            color = 0
        else:
            
            color = 1
    elif len(bands)==1:
        color = 0
    else:
        color = 2
        
    return color

### Removes colored grid from image ###
#----------------------------------------------------------------------#

def color_filter(image):
    # # Automatic Image Thresholding
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    image = img[:,:,2]
    
    return image

### Creates Kernel ###
#----------------------------------------------------------------------#

def makeKernel(size):
    kernel = np.ones((size,11), np.uint8)
    
    return kernel 

### Erosion Image Filtering Technique ###
#----------------------------------------------------------------------#

def Erodes(image, numOfIterations, kernelType):
    
    ErodingImage = cv2.erode(image, kernelType, iterations = numOfIterations)
    
    return ErodingImage

### Dilates Image Filtering Technique ###
#----------------------------------------------------------------------#

def Dilates(image, numOfIterations, kernelType):
    
    Dilated = cv2.dilate(image, kernelType, iterations = numOfIterations)
    
    return Dilated

### Filters Grayscale images ###
#----------------------------------------------------------------------#

def gray_filter(croppedImg):

    kernel = makeKernel(9)
    er1 = Erodes(croppedImg,1,kernel)
    di1 = Dilates(er1,2,kernel)
    er2 = Erodes(di1,2,kernel)
    di2 = Dilates(er2,1,kernel)
    bitNot = cv2.bitwise_not(di2)
    bitAnd = cv2.bitwise_and(croppedImg, bitNot)
    bitAnd = cv2.bitwise_not(bitAnd)
    blu = cv2.blur(bitAnd,(9,9))
    img = cv2.cvtColor(blu, cv2.COLOR_BGR2GRAY)
    
    return img

### Automatic Thresholding for Image Binarization ###
#----------------------------------------------------------------------#

def Otsu(image):
    # Automatic Image Thresholding
    # Otsu's Method
    root.destroy()
    # set booleans
    is_reduce_noise = False
    is_normalized = False

    # read image in grayscale and gaussian blur
    # Apply GaussianBlur to reduce image noise if it is required
    if is_reduce_noise:
        image = cv2.GaussianBlur(image, (5, 5), 0)

    # Calculate Otsu's Threshold
    # Set total number of bins in the histogram
    bins_num = 256

    # Get the image histogram
    hist, bin_edges = np.histogram(image, bins=bins_num)

    # Get normalized histogram if it is required
    if is_normalized:
        hist = np.divide(hist.ravel(), hist.max())

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)

    threshold = bin_mids[:-1][index_of_max_val]

    global image_result
    image_result = (image <= threshold) * 255
    
    return image_result
    
### Creates Brightness Contrast adjustment bars ###
#----------------------------------------------------------------------#

def BrightnessContrast(brightness=0):
     
    # getTrackbarPos returns the current
    # position of the specified trackbar.
    brightness = cv2.getTrackbarPos('Brightness',
                                    'GEEK')
    
      
    contrast = cv2.getTrackbarPos('Contrast',
                                  'GEEK')
    effect = controller(img, brightness, 
                        contrast)
  
    # The function imshow displays an image
    # in the specified window
    cv2.imshow('GEEK', effect)
    
    global image_result
    image_result = (1-effect/255)*255
    
    return effect

### Creates brightness and contrast adjustment values ###
#----------------------------------------------------------------------#
  
def controller(img, brightness=255,
               contrast=127):

    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))

    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
  
    if brightness != 0:
  
        if brightness > 0:
  
            shadow = brightness
  
            max = 255
  
        else:
  
            shadow = 0
            max = 255 + brightness
  
        al_pha = (max - shadow) / 255
        ga_mma = shadow
  
        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(img, al_pha, 
                              img, 0, ga_mma)
  
    else:
        cal = img
  
    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)
  
        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(cal, Alpha, 
                              cal, 0, Gamma)
  
    return cal

### Adjusts contrast and brightness within image window ###
#----------------------------------------------------------------------#

def adjustContrast(img):
    
    root.destroy()
    
    if __name__ == '__main__':

        cv2.namedWindow('GEEK',cv2.WINDOW_NORMAL)
    
        cv2.imshow('GEEK', img)
        
        cv2.createTrackbar('Brightness','GEEK', 255, 2 * 255, BrightnessContrast) 
    
        cv2.createTrackbar('Contrast', 'GEEK', 127, 2 * 127, BrightnessContrast)
  
      
        BrightnessContrast(0)
  
    cv2.waitKey(0)  
    
    
def zoom1(img):
    
    cv2.createTrackbar('Zoom',"Results", 1, 20, zoom2) 
    cv2.createTrackbar('X Position',"Results", 0, 100, zoom2) 
    cv2.createTrackbar('Y Position',"Results", 0, 100, zoom2) 
    
    zoom2(1)
    
    cv2.waitKey(0)  
    
def zoom2(zoomLevel= 1):
    
    zoomLevel = cv2.getTrackbarPos('Zoom',"Results")
    x = cv2.getTrackbarPos('X Position',"Results")
    y = cv2.getTrackbarPos('Y Position',"Results")
   
    if zoomLevel == 0:
        zoomLevel = 1
    
    h = np.size(img,0)
    wi = np.size(img,1)
    X = int((wi-wi/zoomLevel)/100 * x)
    Y = int((h-h/zoomLevel)/100 * y)
    im = img[Y:int(h/zoomLevel)+Y,X:int(wi/zoomLevel)+X]
    cv2.imshow("Results", im)

    
### Determines the number of pixels per millimeter ###
#----------------------------------------------------------------------#

def PixelCount(Image, threshold,window,k):   # ONLY NEED TO CALL THIS ONE
    
    if k == 0:
        window.destroy()
        
    Region = RegionOfInterest(Image)
    Quarter = ImageQuarter(Image)
    
    Starter1 = 1
    Starter2 = 0
    
    if threshold == 0:
        threshold1 = 0.4
        threshold2 = 0.3

    if threshold == 1:
        threshold1 = 0.6
        threshold2 = 0.5

    if threshold == 2:
        threshold1 = 0.7
        threshold2 = 0.6
        
    Continue = True
    
    
    while Continue:
    
        Pointx, Pointy = FindFirstPoint(Quarter, Region, Starter1, Starter2, threshold1)
        
        global pixmm
        pixmm = FindSecondPoint(Quarter, Region, Pointx, Pointy, threshold2)
        
        if pixmm == 0 :
            Starter1 = Pointx + 1
            
        else:
            Continue = False
            
def stdPixelCount(Image, threshold, window):
    k = 0 
    PixelCount(Image, threshold, window,k)
    if pixmm <=3:
        threshold = 1
        window = 0
        k=1
        PixelCount(Image, threshold, window,k)
        if pixmm <= 3:
            threshold = 0
            window = 0
            PixelCount(Image, threshold, window,k)
            
    
### Detects Region within Image ###
#----------------------------------------------------------------------#
    
def AutomatedRegionOfInterest(image):

    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    image_width, image_height = grey_image.shape 
    
### Detects grid region that is applied to calibrate ###
#----------------------------------------------------------------------#

def RegionOfInterest(image):
    
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    image_width, image_height = grey_image.shape 
    
   
    Region_Upper = int(image_height * 0.4)
    Region_Lower = int(image_height * 0.45)
    Region_Left  = int(image_width * 0.4)
    Region_Right = int(image_width * 0.45)
    
    
    Region = grey_image[Region_Upper:Region_Lower , Region_Left: Region_Right]
    
    
    return Region
    
### Finds image quarters for callibration ###
#----------------------------------------------------------------------#

def ImageQuarter(image):

    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_width, image_height = grey_image.shape 
    
      
    Quarter_Lower = int(image_height * 0.5) #Original 0.25
    Quarter_Right = int(image_width * 0.5) #Original 0.25
    
    Quarter_Upper = int(image_height * 0.35)
    Quarter_Left = int(image_width * 0.35)
    
    Quarter = grey_image[Quarter_Upper:Quarter_Lower , Quarter_Left: Quarter_Right]
    
    
    return Quarter
    
### Finds first grid line for calibration ###
#----------------------------------------------------------------------#

def FindFirstPoint(Quarter, Region, Startx, Starty,threshold):
           
    Detect = cv2.matchTemplate(Quarter, Region, cv2.TM_CCOEFF_NORMED)
    
    breaker = False
    
    for iteration1 in range(Startx, len(Detect)):
        for Pixel1 in range(Starty, len(Detect[iteration1])):
            
            if Detect[iteration1][Pixel1] >= threshold:
               
                Pointx = iteration1
                Pointy = Pixel1
                
                breaker = True 
                
                break 
            
        if breaker:
            break
        
    return Pointx, Pointy
    
### Finds second line to complete calibration ###
#----------------------------------------------------------------------#
    
def FindSecondPoint(Quarter, Region, Pointx, Pointy, threshold2):
    
        Detect = cv2.matchTemplate(Quarter, Region, cv2.TM_CCOEFF_NORMED)
        
        breaker = False 
        
        Count = 3

        for Pixel in range(Pointy + 3, len(Detect[0]) ):
            
            if Detect[Pointx][Pixel] >= threshold2 and Count < len(Detect[0]) / 3:
                
                breaker = True
                
                break
                
            Count += 1
            
        if breaker == False:
            Count = 0
            
        return Count

### Determines RR Interval ###
#----------------------------------------------------------------------#

def RRInterval(image,pixmm,mms):

    colSum = sum(image,1)
    
    Rwave = colSum > 1/3 * max(colSum)
    Rwave = Rwave * 1

    pixInt = np.where(Rwave == 1)
    pixInt= np.asarray(pixInt)
    pixInt = pixInt[0][:]
    RRpixel = []
    pixDis = []
     
    for i in range(1,(len(pixInt)-1)):
        pixDis = pixInt[i]-pixInt[i-1]
        if pixDis > 0.1 * pixmm * mms:
            RRpixel.append(pixDis)
        
    RRpixel = np.asarray(RRpixel)
    RRpix = []
    
    for j in range(len(RRpixel)):
        if RRpixel[j] > max(RRpixel)/2:
            RRpix.append(RRpixel[j])
            
    if len(RRpix) == 0:
        RRInt = -1
    else:
        RRpix = np.int_(RRpix)
        q1 = np.quantile(RRpix,0.35)
        q3 = np.quantile(RRpix,0.65)
        low = q1 - 1.5 * (q3 - q1)
        up =  q1 + 1.5 * (q3 - q1)
    
        RR = RRpix[(RRpix>=low) & (RRpix<=up)]
        aveRRpix = mean(RR)
        RRInt = aveRRpix / pixmm / mms
    
    return RRInt

### Determines the distance between two mouse clicks ###
#----------------------------------------------------------------------#

def click_event(event, x, y, flags, params):
    
    global count
    global imgClickTemp
    
    if count == 0:
        imgClickTemp = img.copy()
    
    # checking for right mouse clicks    
    if event==cv2.EVENT_LBUTTONDOWN and count == 1:
        imgClickTemp[y-15:y+15,x-15:x+15] = 255
        imgClickTemp[y-10:y+10, x-10:x+10] = 0
        cv2.imshow("Manual Results", imgClickTemp)
        count = count + 1
        global x2
        x2 = x
        global y2
        y2 = y
  
    if event==cv2.EVENT_LBUTTONDOWN and count == 0:
        imgClickTemp[y-15:y+15,x-15:x+15] = 255
        imgClickTemp[y-10:y+10, x-10:x+10] = 0
        cv2.imshow("Manual Results", imgClickTemp)
        count = count + 1
        global x1
        x1 = x
        global y1
        y1 = y
    
    if event==cv2.EVENT_LBUTTONDOWN and count == 3:
        cv2.imshow("Manual Results", img)
        count = 0
        
    if count == 2:
        distanceHor = abs(x2-x1)/pixmm/25
        distanceVer = abs(y2-y1)/pixmm/10
        img2 = cv2.putText(imgClickTemp, 'Measured Interval:{:.3f} [s], {:.3f} [mV]'.format(distanceHor,distanceVer), (int(len(imgCropped)/2), 150),cv2.FONT_HERSHEY_SIMPLEX,  int(np.size(imgCropped,1)*0.0008), (255, 255, 255),  int(np.size(imgCropped,1)*0.0008))
        cv2.imshow("Manual Results", img2)
        count = count + 1
        
### Creates window with results displayed ###
#----------------------------------------------------------------------#

def ResultsWin(img):
    cv2.namedWindow("Results",cv2.WINDOW_NORMAL)
    cv2.imshow("Results",img)
    zoom1(img)
    cv2.waitKey(0)

### Determines the distance between two mouse clicks ###
#----------------------------------------------------------------------#

def click_event1(event, x, y, flags, params):
    
    global count
    global imgClickTemp
    
    if count == 0:
        imgClickTemp = imgCropped.copy()
    
    # checking for right mouse clicks    
    if event==cv2.EVENT_LBUTTONDOWN and count == 1:
        imgClickTemp[y-15:y+15,x-15:x+15] = [255,0,0]
        cv2.imshow("Original Image Manipulation", imgClickTemp)
        count = count + 1
        global x2
        x2 = x
        global y2
        y2 = y
  
    if event==cv2.EVENT_LBUTTONDOWN and count == 0:
        imgClickTemp[y-15:y+15,x-15:x+15] = [255,0,0]
        cv2.imshow("Original Image Manipulation", imgClickTemp)
        count = count + 1
        global x1
        x1 = x
        global y1
        y1 = y
    
    if event==cv2.EVENT_LBUTTONDOWN and count == 3:
        cv2.imshow("Original Image Manipulation", imgCropped)
        count = 0
        
    if count == 2:
        distanceHor = abs(x2-x1)/pixmm/25
        distanceVer = abs(y2-y1)/pixmm/10
        img2 = cv2.putText(imgClickTemp, 'Measured Interval:{:.3f} [s], {:.3f} [mV]'.format(distanceHor,distanceVer), (int(len(imgCropped)/2), 150),cv2.FONT_HERSHEY_SIMPLEX, int(np.size(imgCropped,1)*0.0008), (255, 0, 0), int(np.size(imgCropped,1)*0.0008))
        cv2.imshow("Original Image Manipulation", img2)
        count = count + 1
        
### Creates window with results displayed ###
#----------------------------------------------------------------------#

def OriginalWin(img1):
    cv2.namedWindow("Original Image",cv2.WINDOW_NORMAL)
    cv2.imshow("Original Image",img1)
    cv2.waitKey(0)

### Displays manual measurements ###
#----------------------------------------------------------------------#

def ManualMeasurements(img):
    cv2.namedWindow("Manual Results",cv2.WINDOW_NORMAL)
    cv2.imshow("Manual Results",img)
    cv2.setMouseCallback("Manual Results", click_event)
    cv2.waitKey(0)


### Displays manual measurements ###
#----------------------------------------------------------------------#

def ManualMeasurements1(imgCropped):
    cv2.namedWindow("Original Image Manipulation",cv2.WINDOW_NORMAL)
    cv2.imshow("Original Image Manipulation",imgCropped)
    cv2.setMouseCallback("Original Image Manipulation", click_event1)
    cv2.waitKey(0)

### Saves un processed image as results ###
#----------------------------------------------------------------------#

def Pointless(img):
    
    img1 = img.copy()
    img2 = img.copy()
    
    root = Tk()
    frm = ttk.Frame(root, padding=50)
    
    frm.grid()
    ttk.Label(frm, text="Final Graphical Results").grid(column=0, row=0)
    ttk.Button(frm, text="Results", command=lambda:ResultsWin(img1)).grid(column=0, row=1)
    ttk.Button(frm, text="Manual Measurements", command=lambda:ManualMeasurements(img2)).grid(column=0, row=2)
    ttk.Button(frm, text="Return", command=root.destroy).grid(column=0, row=3)
    
    mainloop()

### Saves un processed image as results ###
#----------------------------------------------------------------------#

def Pointless1(imgCropped):
     
    img3 = imgCropped.copy()
    img4 = imgCropped.copy()
    
    root = Tk()
    frm = ttk.Frame(root, padding=50)
    
    frm.grid()
    ttk.Label(frm, text="Original Image").grid(column=0, row=0)
    ttk.Button(frm, text="Original Image", command=lambda:OriginalWin(img3)).grid(column=0, row=1)
    ttk.Button(frm, text="Manual Measurements", command=lambda:ManualMeasurements1(img4)).grid(column=0, row=2)
    ttk.Button(frm, text="Return", command=root.destroy).grid(column=0, row=3)
    
    mainloop()
### Allows for adjustment of filtering threshold ###
#----------------------------------------------------------------------#
    
def FilterLevels(img):
    
    root.destroy()
    
    win = Tk()
    frm = ttk.Frame(win, padding=50)

    frm.grid()
    ttk.Label(frm, text="Filter Threshold").grid(column=0, row=0)
    ttk.Button(frm, text="Low Threshold", command=lambda:PixelCount(img,0,win,0)).grid(column=0, row=1)
    ttk.Button(frm, text="Medium Threshold", command=lambda:PixelCount(img,1,win,0)).grid(column=0, row=2)
    ttk.Button(frm, text="High Threshold ", command=lambda:PixelCount(img,2,win,0)).grid(column=0, row=3)
    
    mainloop()

### Updates loop to end while loop ###
#----------------------------------------------------------------------#

def ctUpdate(ans):
    global ct
    if ans !=0:
        ct = 1
        root.destroy()
    else:
        root.destroy()
        
### Main Coding Function ###
#----------------------------------------------------------------------#
    
while(ct == 0):
    random = 0
    
    root = Tk()
    frm = ttk.Frame(root, padding=50)
    frm.grid()
    ttk.Label(frm, text="Image Upload").grid(column=0, row=0)
    ttk.Button(frm, text="Select File", command=openFile).grid(column=0, row=1)
    ttk.Button(frm, text="Quit", command=lambda:ctUpdate(1)).grid(column=0, row=2)
    mainloop()
    
    if ct == 1:
        break
    
    img = cv2.imread(file)
    imgCropped = cropImg(file)   
    color = detect_color_image(file)
    
    root = Tk()
    frm = ttk.Frame(root, padding=50)

    frm.grid()
    ttk.Label(frm, text="Filter Threshold").grid(column=0, row=0)
    ttk.Button(frm, text="Standard Thresholding", command=lambda:stdPixelCount(img,2,root)).grid(column=0, row=1)
    ttk.Button(frm, text="Advanced Options", command=lambda:FilterLevels(img)).grid(column=0, row=2)
    ttk.Button(frm, text="Quit", command=lambda:ctUpdate(1)).grid(column=0, row=3)
    mainloop()

    if ct == 1:
        break
    
    if color == 0:
        img = gray_filter(imgCropped)
    elif color == 1:
        img = color_filter(imgCropped)
    elif color == 2:
        print("Color Detection Error")
    
    root = Tk()
    frm = ttk.Frame(root, padding=50)
    
    frm.grid()
    ttk.Label(frm, text="Image Modification").grid(column=0, row=0)
    ttk.Button(frm, text="Automatic Filtering", command=lambda:Otsu(img)).grid(column=0, row=2)
    ttk.Button(frm, text="Manual Filtering", command=lambda:adjustContrast(img)).grid(column=0, row=3)
    ttk.Button(frm, text="Quit", command=lambda:ctUpdate(1)).grid(column=0, row=4)

    mainloop()
    
    if ct == 1:
        break

    if random == 0:
        RRint = RRInterval(image_result,pixmm,25)
    
    else:
        RRint = -1
    
    img = image_result.astype(np.uint8)
    img = cv2.putText(img, 'RR Interval:{:.3f} [s]'.format(RRint), (150, 150),cv2.FONT_HERSHEY_SIMPLEX, int(np.size(imgCropped,1)*0.0008), (255, 255, 255), int(np.size(imgCropped,1)*0.0008))
    
    root = Tk()
    frm = ttk.Frame(root, padding=50)
    
    frm.grid()
    ttk.Label(frm, text="Final Graphical Results").grid(column=0, row=0)
    ttk.Button(frm, text="Original Image", command=lambda:Pointless1(imgCropped)).grid(column=0, row=1)
    ttk.Button(frm, text="Results", command=lambda:Pointless(img)).grid(column=0, row=2)
    ttk.Button(frm, text="Return", command=root.destroy).grid(column=0, row=3)
    
    mainloop()

    root = Tk()
    frm = ttk.Frame(root, padding=50)
    
    frm.grid()
    ttk.Label(frm, text="Start Over?").grid(column=0, row=0)
    ttk.Button(frm, text="New Image", command=lambda:ctUpdate(0)).grid(column=0, row=1)
    ttk.Button(frm, text="Quit", command=lambda:ctUpdate(1)).grid(column=0, row=2)

    mainloop()
