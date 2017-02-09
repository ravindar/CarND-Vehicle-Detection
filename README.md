##Writeup
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./vehicles/KITTI_extracted/1.png
[image1_a]: ./non-vehicles/GTI/image1.png
[image2]: hog_image.png
[image3]: sliding_windows.png
[image4]: sliding_window_pipeline.png
[image5]: heatmap1.png
[image6]: heatmap2.png
[image7]: heatmap3.png
[image8]: heatmap4.png
[image9]: heatmap5.png
[image10]: heatmap6.png
[training_method]: training_method.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

---
###Writeup / README
#### Helper moethods like slide_window, extract_features, color_hist etc. are in helper_functions.py and P5.ipynb contains the final solution with the pipeline.
###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the `helper_function.py`

```
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features
```

There are two parts to this function. The above function uses `from skimage.feature import hog` to extract features.
The function takes in orientation, pixel per cells and cells per block to get extracted features. You can also pass in a variable `vis` which when `True` will also return the hog image

The above method `get_hog_features` is called from `extract_features`
```
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features
```

The method above `extract_features` also does spatial binning and also does a color histogram.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Vehicle Image][image1] ![Non-Vehicle Image][image1_a]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HLS` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and example:
```
    `orientations=16`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`
    `orientations=10`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`
    `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`
```

The winning one was
```
    `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`
```

Repeating the `extract_feature` method from above

```
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features
```


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using. You can see the code in Section `Running SVC` in P5.ipynb
Here is what I did to run the SVC on a sample test_image

1. Read an image
2. get sliding_windows of varing sizes (96, 96) with a 85% overlap and (128, 128) with a 75% overlap.
3. search for vehicles in the sliding windows from step 2 above using LinearSVC trained above
    3.1. I used these values color_space -> 'YCrCb', HOG orientations -> 9, HOG pixels per cell -> 2, HOG Channel -> 'ALL'
    3.2. Number of histogram bins -> 32  and Spatial binning of (32, 32)
4. After I get positive car images I use heat mapping to identify cars
5. Finally I apply a threshold to remove false positives

![Training code][training_method]

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

As described above I used sliding_windows of varing sizes `(96, 96)` with a `85%` overlap and `(128, 128)` with a `75%` overlap. The sliding window is only run on half the image. The larger sliding window is run from y_start_stop=[360, 700] and the smaller sliding window is run from y_start_stop=[360, 500]. The smaller window is only run  y_start_stop=[360, 500] coz it is used to detect cars further away and appear smaller. The code that generates the sliding windows is below.
```
# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan / nx_pix_per_step) - 1
    ny_windows = np.int(yspan / ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list
```

![alt text][image3]

The code that generated the above sliding window, heat-map and final box drawn image is below
```
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    bboxes =[]
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox)
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img, bboxes

def add_heat(image, bbox_list):
    #print(image.shape[0], image.shape[1])
    heatmap = np.zeros_like(image[:,:,0]).astype(np.float32)
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
#         print(box[0][1], box[1][1], box[0][0], box[1][0])
#         print(heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]])

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

draw_image = np.copy(test_image)

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
test_image = test_image.astype(np.float32)/255

# small_windows = slide_window(test_image, x_start_stop=[None, None], y_start_stop=[360, 500],
#                        xy_window=(64, 64), xy_overlap=(0.50, 0.50))
small_windows = slide_window(test_image, x_start_stop=[800, None], y_start_stop=[360, 500],
                           xy_window=(96, 96), xy_overlap=(0.85, 0.85))

# medium_windows = slide_window(test_image, x_start_stop=[None, None], y_start_stop=[400, 600],
#                        xy_window=(96, 96), xy_overlap=(0.50, 0.50))

large_windows = slide_window(test_image, x_start_stop=[None, None], y_start_stop=[360, 700],
                       xy_window=(128, 128), xy_overlap=(0.75, 0.75))

windows = []
for w in small_windows:
    windows.append(w)

# for w in medium_windows:
#     windows.append(w)

for w in large_windows:
    windows.append(w)

search_window_image = draw_boxes(draw_image, windows)

hot_windows = search_windows(test_image, windows, svc, X_scaler, color_space=color_space,
                             spatial_size=spatial_size, hist_bins=hist_bins,
                             orient=orient, pix_per_cell=pix_per_cell,
                             cell_per_block=cell_per_block,
                             hog_channel=hog_channel, spatial_feat=spatial_feat,
                             hist_feat=hist_feat, hog_feat=hog_feat)

hot_windows_image = draw_boxes(draw_image, hot_windows)

heat_map_image = add_heat(draw_image, hot_windows)
thresholded_image = apply_threshold(heat_map_image, 2)

from scipy.ndimage.measurements import label

labels = label(thresholded_image)
print(labels[1])

window_img, bbox_list = draw_labeled_bboxes(draw_image, labels)

```

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I tried RGB with 3-channel HOG. The best test accuracy i could get was
```
Test Accuracy of SVC =  0.97
```

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.
The test accuracy imrpoved to
```
Test Accuracy of SVC =  0.9901
```

Here are some example images:

![alt text][image4]

The Pipeline code is in `Creating Pipeline` section of `P5.ipynb`

-----------------------------------------------------------------

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a (project_video_output.mp4) [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

The code that build heatmaps and does threshold is below
```
def add_heat(image, bbox_list):
    #print(image.shape[0], image.shape[1])
    heatmap = np.zeros_like(image[:,:,0]).astype(np.float32)
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
#         print(box[0][1], box[1][1], box[0][0], box[1][0])
#         print(heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]])

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

```

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps, bounding box:
#### Image 1
![alt text][image5]

#### Image 2
![alt text][image6]

#### Image 3
![alt text][image7]

#### Image 4
![alt text][image8]

#### Image 5
![alt text][image9]

#### Image 6
![alt text][image10]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The approach is I took was to do
1. load vehicle and non-vehicle data
2. extract features based on spatial binning, HOG and color histograming
3. getting training and test data and running a LinearSVC
4. Writing code to run sliding windows(varying sizes and on only half the image - smaller window is y_start_stop=[360, 500] and larger window is  y_start_stop=[360, 700]) and extract features.
5. I run the classifiers on these windows from step 4. For all the images where there is vehicle I mark it as a hot window
6. I run heat mapping to essentially verify vehicles where overlapping windows have positive vehicles
7. I apply threshold to remove false positives
8. Run the above pipeline on video

#### Issues
The biggest issue was finding the HOG parameters. It seems like I had to run many options to figure out what worked.
There was a similar issue with finding the color space that worked. I tried RGB and had a 97% test accuracy, which I felt was good enough. But when I used that classifier I found that the pipeline had a lot of false positives and the vehicle detection was not good.
The color space that worked was YCrCb. But that was after I tried a bunch of color spaces.

The pipeline was built completely using HOG, color histogram, spatial binning etc. I felt this is not a very flexible pipeline and has a heavy dependence on image features. Also the classification is based on a small data set provided by the class (`vehicle` and `non-vehicle` folder). I feel I could use a larger pool of images to help with classification. Also the pipeline could be built using a convelutional network which would be a lot more flexible during classification.

