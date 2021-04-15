import os
import numpy as np
import json
from PIL import Image

''' Helper functions for running predictions. '''
def compute_convolution(I, T, stride=1, window_size=None, padding=None):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''
    (n_rows, n_cols, n_channels) = np.shape(I)
    
    '''
    BEGIN YOUR CODE
    '''
    # Get shape of template
    (t_rows, t_cols, t_channels) = np.shape(T)
    
    # Add padding to preserve shape of image
    x_pad, y_pad = t_rows-1, t_cols-1
    I = np.pad(I, ((x_pad, x_pad), (y_pad, y_pad), (0,0)), 'constant') # ((top, bot),(left,right))
    
    heatmap = np.zeros((n_rows, n_cols))
    for row in range(0, n_rows, stride):
        for col in range(0, n_cols, stride):
            conv = np.sum(T * I[row:row+t_rows,col:col+t_cols]) / ((t_rows+1)*(t_cols+1)) # * 255
            heatmap[row, col] = conv

    '''
    END YOUR CODE
    '''
    return heatmap

def predict_boxes(I, heatmap, threshold, stride=1, box_size=10):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    '''
    BEGIN YOUR CODE
    '''
    (n_rows, n_cols, n_channels) = np.shape(I)
    (h_rows, h_cols) = np.shape(heatmap)
    
    # Compute size of boxes based on image and heatmap
    box_height = box_size
    box_width = box_size

    print("Box size = ", box_size)
    num_boxes = 0
    for row in range(0, h_rows, stride):
        for col in range(0, h_cols, stride):
            ul_x, ul_y = row, col
            br_x, br_y = row + box_height, col + box_width
            score = heatmap[row, col]
            if score  > threshold:
                flag = True
                # Check that bounding boxes do not overlap
                for [x1, y1, x2, y2, s] in output:
                    dist1 = np.sqrt((x1-ul_x)**2 + (y1-ul_y)**2)
                    dist2 = np.sqrt((x2-ul_x)**2 + (y2-ul_y)**2)
                    if dist1 < 20 or dist2 < 20:
                        flag = False
                mean_flux = np.mean(I[ul_x:br_x, ul_y:br_y])
                if box_size > 3 and (mean_flux > 1.1*0.105):
                    flag = False
                if box_size == 3 and (mean_flux > 1.05*0.105 or mean_flux < 0.9*0.105):
                    flag = False
                
                if flag:
                    print("Avg Pixel Flux = ", np.mean(I[ul_x:br_x, ul_y:br_y]))
                    output.append([ul_x, ul_y, br_x, br_y, score])
                    num_boxes += 1

    #print("Num Boxes Predicted for threshold ", threshold, " = ", num_boxes)
    '''
    END YOUR CODE
    '''

    return output

def normalize_image(img, kernel):
    '''
    For an image patch or kernel, img, normalize it with respect to each color channel using
    the mean of each color channel of the full image, I. Return the normalized img.
    '''
    img_normalized = img.copy().astype(np.float)
    r, g, b = np.mean(kernel[:, :, 0]), np.mean(kernel[:, :, 1]), np.mean(kernel[:, :, 2])

    img_normalized[:, :, 0] = 3 * np.maximum(np.subtract(img_normalized[:, :, 0], r), 0) / 255
    img_normalized[:, :, 1] = 0.1*np.maximum(np.subtract(img_normalized[:, :, 1], g), 0) / 255
    img_normalized[:, :, 2] = 0.1*np.maximum(np.subtract(img_normalized[:, :, 2], b), 0) / 255
        
    return img_normalized 

def detect_red_light_mf(I, stride=1):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''
    # You may use multiple stages and combine the results
    T = Image.open('kernel.png').convert(mode='RGB') # base kernel
    I_normalized = normalize_image(I, np.asarray(T))
    I = I_normalized
    
    output = []
    thresholds = [0.264, 0.435, 0.62, 0.63] # 0.264,
    T_sizes = [3, 5]
    for i in range(len(T_sizes)):
        T_size, threshold = T_sizes[i], thresholds[i]
        T_resized = T.copy().resize((T_size, T_size))
        T_resized = np.asarray(T_resized)
        T_normalized = normalize_image(T_resized, T_resized)
        
        T_resized = T_normalized
        
        heatmap = compute_convolution(I, T_resized, stride)
        predicted_boxes = predict_boxes(I, heatmap, threshold, stride, T_size)
        output.extend(predicted_boxes)

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        if (output[i][4] < 0.0) or (output[i][4] > 1.0):
            print(output[i][4])
        
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

# Visualize predictions
def visualize_bounding_boxes(I, bounding_boxes):
    '''
    This function takes a numpy image arry <I> and a list of
    <bounding_boxes> and displays I with bounding boxes. 
    Each element of <bounding boxes> is a 4-integer list 
    specifying the top left and bottom right corners of bounding 
    boxes contained in the image I. 
    '''
    fig, ax = plt.subplots()
    plt.imshow(I)
    idx = 0
    for [ul_x, ul_y, br_x, br_y, score] in bounding_boxes:
        (x, y) = (ul_x, ul_y)
        rect = matplotlib.patches.Rectangle((y, x), -abs(ul_x - br_x), -abs(br_y - ul_y),
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(y, x, str(round(score, 4)), color='white')
        idx += 1
        
    plt.show()
    
''' Run predictions! '''
# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '/Users/Johanna/Desktop/Computer Vision/hw01/RedLights2011_Medium'

# load splits: 
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '../data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

'''
Make predictions on the training set.
'''
preds_train = {}
for i in range(len(file_names_train)):
    print("Image: ", i+1)
    # read image
    img = Image.open(os.path.join(data_path,file_names_train[i]))
    #img = Image.open('small_image.png').convert('RGB')
    img = np.asarray(img)

    # detect red lights in image
    preds_train[file_names_train[i]] = detect_red_light_mf(img, stride=1)
    # show bounding boxes
    visualize_bounding_boxes(img, preds_train[file_names_train[i]])
    
# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))
        
        # convert to numpy array:
        I = np.asarray(I)
        
        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)