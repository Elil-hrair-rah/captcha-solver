import numpy as np
import discord
from discord.ext import commands
import time
import random
import idx2numpy
import matplotlib.pyplot as plt
import cv2
import pytesseract
import math
import sys
from PIL import Image
import requests
from io import BytesIO
#from scipy.signal import find_peaks


pytesseract.pytesseract.tesseract_cmd = "F:/Tesseract-OCR/tesseract.exe"

#parallel processing package for performing computations with gpu
#helps with optimization and avoiding errors
#code appears to run fine without it
#import numba as nb

class Group:
    def __init__(self, start, end):
        self.start = start
        self.end = end
    
    def __abs__(self):
        return self.end - self.start
    
    def thirds(self):
        offset = abs(self) // 3
        return Group(self.start, self.start + offset), \
               Group(self.start + offset, self.end - offset), \
               Group(self.end - offset, self.end)
    
    def halves(self):
        offset = abs(self) // 2
        return Group(self.start, self.start + offset), \
               Group(self.start + offset, self.end)

#use seed for consistent results
#np.random.seed(12)

def traindata():
    #import data from idx file
    
    #since idx2numpy is not available on school computers, use npy files instead
    ndarr = idx2numpy.convert_from_file('train-images.idx3-ubyte')
    results = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
    
    #normalize data
    ndarr = ndarr/255.0
    #flatten image array for a 900x1 vector rather than a 30x30 matrix
    x = []
    for i in ndarr:
        i = np.append(i, [np.zeros(28)]*2, 0)
        i = np.append(i, [np.zeros(2)]*30, 1)
        x.append(i.flatten())
    x = np.asarray(x)
    #convert from numerical label to array-like label for better fitting
    y = []
    for i in results:
        zeros = np.zeros(10)
        zeros[i] = 1
        y.append(zeros)
    y = np.asarray(y)
    return x, y
    
def testdata():
    #import data from idx file
    
    #since idx2numpy is not available on school computers, use npy files instead
    ndarr = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
    results = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')
    
    
    #normalize data
    ndarr = ndarr/255.0
    x = []
    #flatten image array for a 900x1 vector rather than a 30x30 matrix
    for i in ndarr:
        i = np.append(i, [np.zeros(28)]*2, 0)
        i = np.append(i, [np.zeros(2)]*30, 1)
        x.append(i.flatten())
    x = np.asarray(x)
    
    y = []
    for i in results:
        zeros = np.zeros(10)
        zeros[i] = 1
        y.append(zeros)
    y = np.asarray(y)
    #convert from numerical label to array-like label for better fitting
    return x, y

def unflatten(data):
    count = 0
    x = []
    for i in range(30):
        y = []
        for j in range(30):
            y.append(data[count])
            count += 1
        x.append(y)
    return np.asarray(x)

#sigmoid function
#uses numba and cuda to (potentially) speed up runtime
#can probably comment out "@nb.jit" if the computer doesn't support GPU computation
#however this may potentially result in runtime issues or significant slowdowns
    
#@nb.jit
def nonlin1(x):
    return 1/(1+np.e**(-x))

#@nb.jit
def nonlin2(x,y):
    sigma = 1/(1+np.e**(-x))
    return (sigma*(1-sigma))*y

#@nb.jit
def error(x,y):
    return x - y

#@nb.jit
def error2(x,y):
    return (x - y)**2

def train(n = 200):
    x,y = traindata()
        
    #set initial weights randomly
    #using a 900 x 10 matrix since the desired result is a 10-vector
    
    weights = 2 * np.random.random((900,10)) - 1
    
    for i in range(n):
        print(i)
        delta = []
        AT = []
        
        #dismantle the data array, otherwise 60000 x 900 is too large of an array to
        #being doing matrix operations on
        #this mitigates out of memory errors but runs slower as a result
        for j, _ in enumerate(x):
            #forward propagation
            A = x[j]
            
            B = nonlin1(np.dot(A,weights))
              
            #error check
            B_error = error(y[j],B)
                
            #back propogation
            B_delta = nonlin2(B,B_error)
                
            #reconstruct the matrices
            AT.append(A.T)
            delta.append(B_delta)
                
    #    print('x')
        AT = np.asarray(AT)
        delta= np.asarray(delta)
        
        #check to make sure that the delta is decreasing as the neural network runs
        difference = np.sqrt(np.sum(np.square(delta)))
#        difference = np.sum(delta)
        print(difference)
        
        #update weights
        weights = weights + np.dot(AT.T,delta)
    
            
    #print('weights after training:')
    #print(weights)
    return weights

    
    #check accuracy of training       
def test(weights):
    #unpack test dataset    
    a,b = testdata()
    #count how many correct guesses
    count = 0
    for i in range(10000):
        
        k0 = a[i]
        
        #forward propogation based on learned weights
        k1 = np.dot(k0,weights)
        
        #compare forward propogation results to theoretical results
        index = np.argmax(k1)
        #increment counter if correct
        if index == np.argmax(b[i]):
            count += 1
    
    print(str(count) + ' correct over 10000 trials')

'''    
    #randomly select 10 images to show
    rand = []
    while len(rand) < 10:
        c = np.random.randint(10000)
        if c not in rand:
            rand.append(c)
    
    fignum = 1
    for i in rand:
        image = unflatten(a[i])
        plt.figure(fignum)
        plt.imshow(image,cmap = 'gray_r',vmin = 0,vmax = 1)
        plt.title('label = ' + str(np.argmax(b[i])))
        fignum += 1
        
        #forward propogation based on learned weights
        k1 = np.dot(a[i],weights)
        
        #normalize data using sigmoid function
        #k1 = nonlin1(k1)
        plt.figure(fignum)
        plt.bar(range(10),k1)
        plt.ylabel('confidence')
        plt.title('predicted number = ' + str(np.argmax(k1)))
        fignum += 1
'''

def get_weights():
    try:
        weights = np.loadtxt('weights.txt', delimiter = ',')
    except IOError:
        weights = train()
        list_string = []
        for i in weights:
            list_string.append(','.join(str(x) for x in i))
        string = '\n'.join(list_string)
        f = open('weights.txt', 'w')
        f.write(string)
        f.close()
    return weights

# Evaluate image based on trained weights
def eval_image(image, weights):
    guesses = np.dot(image, weights)
    
    guess = np.argmax(guesses)
    
    '''
    #just manually change it based on the number it looks like most lol
    guess = np.argmax(guesses)
    if guess == 0:
        guess = 8
    if guess == 1:
        guess = 7
    if guess == 9:
        guess = 3
    '''
    
    #if 0, 1, or 9, retry
    guess = 0
    while guess == 0 or guess == 1 or guess == 9:
        guess = np.argmax(guesses)
        guesses[guess] = guesses[np.argmin(guesses)]
    
    return guess

#from https://stackoverflow.com/questions/37745519/use-pytesseract-to-recognize-text-from-image
def image_processing(image):
    # Grayscale, Gaussian blur, Otsu's threshold
    # image = cv2.imread('8.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Morph open to remove noise
    # No need to invert the data, as it matches the black bg/white font of the
    # MNIST data
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Perform some erosion now, before scaling changes
    opening = cv2.erode(opening, np.ones((3,3)), iterations = 1)
    
    #invert = 255 - opening
    invert = opening
    
    
    # Apply scaling changes to image to shrink it to half size
    scale_factor = 0.5
    scaling = (int(invert.shape[1] * scale_factor), int(invert.shape[0] * scale_factor))
    output = cv2.resize(invert, scaling)
    return output

def show_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# Slices the image into 30x30 squares
def image_snip(image):
    divs = []
    for i in range(len(image[0])-30):
        divs.append(image[:,i:i+30])
    return divs

# Sums up the darkness of the image
def pixel_density(image):
    return np.sum(image)

# Attempts to apply tesseract OCR to image
def tess_test(image):
    data = pytesseract.image_to_string(image, lang='eng', config='--psm 12')
    print(data)
    
# Find the first group of an image that are greater than threshold, starting search at init
def threshold(image, thresh, init):
    start = -1
    end = 0
    while start == -1 and init < len(image):
        if image[init] >= thresh:
            start = init
        else:
            init += 1
    while init < len(image) and image[init] >= thresh:
        init += 1
        end = init
    if start == -1:
        return None
    return Group(start, end)

# Find all groups of image with minimum width min_width
def all_threshold(image, thresh):
    groups = []
    result = Group(-1, 0)
    while result:
        result = threshold(image, thresh, result.end)
        if result and result.end - result.start > 5:
            groups.append(result)
    '''
    result = [-1, 0]
    while result != [-1, -1]:
        result = threshold(image, thresh, result[1])
        groups.append(result)
    '''
    return groups

# Scans from thresh_low to thresh_high until target_groups groups have been found.
# Returns the groups if found, returns 0 if not found.
def group_search(image, target_groups, thresh_low, thresh_high):
    found = 0
    while thresh_low < thresh_high:
        if len(all_threshold(image, thresh_low)) == target_groups:
            found = all_threshold(image, thresh_low)
            thresh_low = thresh_high
        else:
            thresh_low += 1
    return found            
    
# Searches for the threshold at which there are maximally many groups of the image.
# Begin at thresh_low, and increase to thresh_high.
# If, at any point, 3 groups are found, end the program.
# Otherwise, look for 2 groups, and then split the larger one in half.
# Otherwise, split the largest group into thirds, at thresh_low
def groups(image, thresh_low, thresh_high):
    if group_search(image, 3, thresh_low, thresh_high) == 0:
        if group_search(image, 2, thresh_low, thresh_high) == 0:
            # If one chunk, divide into thirds and return
            group = all_threshold(image, thresh_low)
            return group[0].thirds()
        # If two chunks, divide the largest chunk in half and return
        group = group_search(image, 2, thresh_low, thresh_high)
        if abs(group[0]) > abs(group[1]):
            return group[0].halves()[0], group[0].halves()[1], group[1]
        else:
            return group[0], group[1].halves()[0], group[1].halves()[0]
    else:
        # If three chunks, return chunks
        return group_search(image, 3, thresh_low, thresh_high)
      
# Assumes the image is taller than it is long
# Will also bias slightly towards right side
def center_image(image):
    target_length = len(image)
    diff = target_length - len(image[0]) 
    left = [np.zeros(math.ceil(diff/2))]*target_length
    right = [np.zeros(math.floor(diff/2))]*target_length
    temp = np.append(left, image, 1)
    image = np.append(temp, right, 1)
    return image

def solve_captcha(image, weights):
    cutoff_min = 400
    cutoff_max = 600
    processed = image_processing(image)
    vert_dens = []
    for i,_ in enumerate(processed[0]):
        vert_dens.append(pixel_density(processed[:,i]))
    plt.figure(100)
    plt.plot(vert_dens)
    threshs = groups(vert_dens,cutoff_min, cutoff_max)
    nums = []
    for i in threshs:
        slim = processed[:,i.start:i.end]
        square = center_image(slim)
        guess = eval_image(square.flatten(), weights)
        # Write all images to the archive 
        global archive_images 
        archive_images = np.append(archive_images, np.array([square.flatten()]), 0)
        nums.append(guess)
    return ''.join([str(x) for x in nums])

def correct_solution(solution):
    global archive_images
    global archive_values
    values = np.array([int(i) for i in solution])
    if len(archive_images) % 3 == 0:
        archive_values = np.append(archive_values, [values] * (len(archive_images) // 3))
        archive_images, archive_values = write_dataset(archive_images, archive_values)
    else:
        archive_images, archive_values = np.empty((0,900)), np.empty((0))
    
def link_solution(link, weights):
    response = requests.get(link)
    img = Image.open(BytesIO(response.content))
    image = np.asarray(img)
    return solve_captcha(image, weights)
    
def js_scrape_to_dataset(image_file = 'training_images.txt', value_file = 'training_values.txt'):
    cutoff_min = 400
    cutoff_max = 600
    images = open(image_file, 'r')
    values = open(value_file, 'r')
    image_array = images.read().split('\n')
    value_array = values.read().split('\n')
    images_array = []
    values_array = []
    for i, a in enumerate(image_array):
        response = requests.get(image_array[i])
        img = Image.open(BytesIO(response.content))
        image = np.asarray(img)
        processed = image_processing(image)
        vert_dens = []
        for j,_ in enumerate(processed[0]):
            vert_dens.append(pixel_density(processed[:,j]))
        print(a)
        threshs = groups(vert_dens, cutoff_min, cutoff_max)
        for j,k in enumerate(threshs):
            slim = processed[:,k.start:k.end]
            square = center_image(slim)
            images_array.append(square.flatten())
            values_array.append(value_array[i][j])
    return images_array, values_array

# Write to a custom image dataset for future training purposes
def write_dataset(array_images, array_nums):
    images,values = read_dataset()
    images = np.append(images, array_images, 0)
    values = np.append(values, array_nums, 0)
    
    list_images = np.array([])
    for i in images:
        list_images = np.append(list_images, ','.join(str(x) for x in i))
    image_string = '\n'.join(list_images)
    value_string = '\n'.join([str(x) for x in values])
    f = open('dataset_image.txt','w') 
    g = open('dataset_value.txt','w')
    f.write(image_string)
    g.write(value_string)
    
    return np.empty((0, 900)), np.empty((0))
    
# Read a custom image dataset for future training
def read_dataset():
    try:
        images = np.loadtxt('dataset_image.txt', delimiter = ',')
        values = np.loadtxt('dataset_value.txt')
    except IOError:
        images = np.empty((0,900))
        values = np.empty((0))
    return images, values

weights = get_weights()

#test(weights)

#x,y = testdata()

archive_images = np.empty((0, 900))
archive_values = np.empty((0))
latest_guess = 0


discordclient = commands.Bot(command_prefix = '!')

@discordclient.event
async def on_message(message):
    channel = discordclient.get_channel(lorem ipsum)
    if message.author.id == lorem ipsum and message.channel.id == lorem ipsum:
        await channel.send(message.content)
    if message.author.id == lorem ipsum and message.channel.id == lorem ipsum:
        if len(message.embeds) > 0:
            embed = message.embeds[0]
            if 'Type in below what you see' in embed.description:
                response = requests.get(embed.image.url)
                img = Image.open(BytesIO(response.content))
                image = np.asarray(img)
                latest_guess = solve_captcha(image, weights)
                await channel.send(latest_guess)
            elif 'Thank you for completing the captcha' in embed.description:
                correct_solution(latest_guess)

discordclient.run("insert token here")
