# Jerret Stovall
# CS 3150
# Midterm Project
# Dr. Feng

import cv2
import matplotlib.pyplot as plt

# Use color mask to isolate foot holds
green_high = (160, 210, 120)
green_low = (80, 130, 30)
yellow_high = (255, 255, 75)
yellow_low = (155, 120, 0)
orange_high = (255, 90, 75)
orange_low = (115, 50, 30)
pink_high = (255, 95, 160)
pink_low = (160, 40, 76)
blue_high = (100, 155, 255)
blue_low = (25, 45, 120)
purple_high = (140, 70, 120)
purple_low = (75, 40, 60)
white_high = (200, 200, 200)
white_low = (150, 150, 150)

color_dictionary = {
    "green"  : (green_high, green_low),
    "yellow" : (yellow_high, yellow_low),
    "orange" : (orange_high, orange_low),
    "pink"   : (pink_high, pink_low),
    "blue"   : (blue_high, blue_low),
    "purple" : (purple_high, purple_low),
}


# This method creates a mask for the different colored foot holds
def detect_foot_holds(img, high_value, low_value):
    h_val = high_value
    l_val = low_value
    mask = cv2.inRange(img, l_val, h_val)
    return cv2.bitwise_and(img, img, mask=mask)


# This method is for opening an image
def open_img(img):
    kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel1)
    return opened


# This method is for closing an image
def close_img(img):
    kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (18, 18))
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel2)
    return closed


# Read input image
src_path = "E:\Old Class Work\CS 3150\Midterm Project\\src_img.JPG"
src_img = cv2.imread(src_path)

# Convert source image from BGR to RGB
src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
plt.figure()
plt.title('Source Image')
plt.imshow(src_img)

# Gamma correction
luv = cv2.cvtColor(src_img, cv2.COLOR_RGB2Luv)
l = luv[:, :, 0]
l = l / 256.0
l = l**.99
l = l * 256
luv[:, :, 0] = l
src_img = cv2.cvtColor(luv, cv2.COLOR_LUV2RGB)

# Detect pink colored foot holds
high, low = color_dictionary['pink']
foot_holds = detect_foot_holds(src_img, high, low)
plt.figure()
plt.title('Pink Foot Holds Detected')
plt.imshow(foot_holds)

# Convert to grey scale
grey_holds = cv2.cvtColor(foot_holds, cv2.COLOR_RGB2GRAY)
grey_holds = cv2.erode(grey_holds, (50, 50), iterations=5)

# Perform morphological transformations
enhanced_img = open_img(grey_holds)
enhanced_img = close_img(enhanced_img)


plt.figure()
plt.title('Enhanced Image To Be Used In Convex Hull Operation')
plt.imshow(enhanced_img, cmap='gray', vmin=0, vmax=255)

# Circle the foot holds
ret, thresh = cv2.threshold(enhanced_img, 0, 255, 0)
contours, hierarchy = cv2.findContours(thresh, 1, 2)
cnt = contours[0:20]
for contour in contours:
    convexHull = cv2.convexHull(contour)
    cv2.drawContours(src_img, [convexHull], -1, (57, 255, 20), 15)

plt.figure()
plt.title('Pink Foot Holds Circled using Convex Hull')
plt.imshow(src_img)
plt.show()

# Read input image again / Cause the one we read in before has the pink foot holds still circled
src_img = cv2.imread(src_path)

# Convert source image from BGR to RGB
src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

# Gamma correction
luv = cv2.cvtColor(src_img, cv2.COLOR_RGB2Luv)
l = luv[:, :, 0]
l = l / 256.0
l = l**.9
l = l * 256
luv[:, :, 0] = l
src_img = cv2.cvtColor(luv, cv2.COLOR_LUV2RGB)

# Detect yellow foot holds
high, low = color_dictionary['yellow']
foot_holds = detect_foot_holds(src_img, high, low)
plt.figure()
plt.title('Yellow Foot Holds Detected')
plt.imshow(foot_holds)

# Convert to grey scale
grey_holds = cv2.cvtColor(foot_holds, cv2.COLOR_RGB2GRAY)
grey_holds = cv2.erode(grey_holds, (3, 3), iterations=1)

# Perform morphological transformations
enhanced_img = open_img(grey_holds)
enhanced_img = close_img(enhanced_img)

plt.figure()
plt.title('Enhanced Image To Be Used In Convex Hull Operation')
plt.imshow(enhanced_img, cmap='gray', vmin=0, vmax=255)

# Circle the foot holds
ret, thresh = cv2.threshold(enhanced_img, 0, 255, 0)
contours, hierarchy = cv2.findContours(thresh, 1, 2)
cnt = contours[0:20]
for contour in contours:
    convexHull = cv2.convexHull(contour)
    cv2.drawContours(src_img, [convexHull], -1, (57, 255, 20), 15)

plt.figure()
plt.title('Yellow Foot Holds Circled using Convex Hull')
plt.imshow(src_img)
plt.show()
