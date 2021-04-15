'''Evaluate model performance. '''
def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    [x1, y1, x2, y2] = box_1
    
    [x3, y3, x4, y4] = box_2

    # Compute area of intersection
    area_intersection = max(0, min(x2, x4) - max(x1, x3) + 1) * max(0, min(y2, y4) - max(y1, y3) + 1)

    # Compute area of union = area_box1 + area_box2 - area_intersection
    area_union = (abs(x1-x2) * abs(y1-y2)) + (abs(x3-x4) * abs(y3-y4)) - area_intersection
    
    # IoU = area of intersection / area of union
    iou = area_intersection / area_union

    assert (iou >= 0) and (iou <= 1.0)
    return iou

# Helper function: visualize ground truth and predicted bounding boxes 
def visualize_gt_and_preds(I, preds, gts,  filename):
    fig, ax = plt.subplots()
    plt.imshow(I)
    preds = preds[filename] # predicted bounding boxes
    for [ul_x, ul_y, br_x, br_y, score] in preds:
        (x, y) = (ul_x, ul_y)
        rect = matplotlib.patches.Rectangle((y, x), -abs(ul_x - br_x), -abs(br_y - ul_y),
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    gt = gts[filename] # predicted ground truth labels
    for [ul_x, ul_y, br_x, br_y] in gt:
        (x, y) = (ul_x, ul_y)
        rect = matplotlib.patches.Rectangle((y, x), abs(ul_x - br_x), abs(br_y - ul_y),
                                 linewidth=1, edgecolor='g', facecolor='none')
        
        max_iou = 0.0
        for pred in preds:
            iou = compute_iou(pred, [ul_x, ul_y, br_x, br_y])
            if iou > max_iou:
                max_iou = iou
        
        plt.text(y, x, str(max_iou), color="white")
        ax.add_patch(rect)
    
    plt.show()
    
def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP, FP, FN = 0, 0, 0

    '''
    BEGIN YOUR CODE
    '''
    # Total ground truth objects
    N =  np.sum([len(gts[file]) for file, pred in preds.items()])
    # Total predictions
    M = np.sum([len(pred) for file, pred in preds.items()]) 
    for pred_file, pred in preds.items():
        gt = gts[pred_file]
        found = False
        for i in range(len(gt)): # Iterate ground truth objects
            for j in range(len(pred)): # Iterate predictions
                iou = compute_iou(pred[j][:4], gt[i])
                confidence = pred[j][4]
                #print(confidence)
                # apply threshold and confidence to determine True Positives (TP)
                if not found and iou > iou_thr and confidence >= conf_thr: 
                    TP += 1
                    found = True
    # False Positives (FP) = Predictions - True Positives (TP)
    FP = M - TP
    # False Negatives (FN) = Ground Truth Objects - True Positives (TP)
    FN = N - TP
    
    assert (TP + FP == M)
    assert (TP + FN == N)
    
    '''
    END YOUR CODE
    '''
    return int(TP), int(FP), int(FN)

''' Get precision and recall values for a given IOU Threshold. '''
def get_pr(iou_thr):
    confidence_thrs = np.linspace(0.0, 0.44, 5)
    tp_train = np.zeros(len(confidence_thrs))
    fp_train = np.zeros(len(confidence_thrs))
    fn_train = np.zeros(len(confidence_thrs))
    precision = []
    recall = []
    for i, conf_thr in enumerate(confidence_thrs):
        tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=iou_thr, conf_thr=conf_thr)

        # Visualize
        '''
        filename = file_names_train[i]
        I = Image.open(os.path.join(data_path,filename))
        I = np.asarray(I)
        visualize_gt_and_preds(I, preds_train, gts_train, filename)
        '''
        # Print results
        #print("Confidence Threshold = ", conf_thr)
        print("TP = ", tp_train[i], ", FP = ", fp_train[i], ", FN = ", fn_train[i])

        precision.append(tp_train[i]/(tp_train[i] + fp_train[i]))
        recall.append(tp_train[i]/(tp_train[i] + fn_train[i]))
    return precision, recall
    
# path for image data
data_path = '/Users/Johanna/Desktop/Computer Vision/hw01/RedLights2011_Medium'

# set a path for predictions and annotations:
preds_path = '../data/hw02_preds'
gts_path = '../data/hw02_annotations'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

'''
Load training data. 
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold. 
    
# Plot training set PR curves
for iou_thr in [0.1, 0.25, 0.3, 0.35, 0.5, 0.75]:
    p, r = get_pr(iou_thr)
    plt.plot(r, p)
    plt.scatter(r, p)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim(0, 0.3)
    plt.title('PR Curve for IOU Threshold of '+ str(iou_thr))
    plt.show()

if done_tweaking:
    print('Code for plotting test set PR curves.')