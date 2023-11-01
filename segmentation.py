import numpy as np
import cv2

def calculate_tumor_area(image_path):
    # Read lung image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    # Noise removal with morphological opening
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground (lung region)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.25 * dist_transform.max(), 255, 0)  # Adjust the 0.25 value 

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Increment all markers to ensure background is not 0
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply Watershed
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    im1 = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    # Get the segmented tumor region
    tumor_mask = (markers == -1)

    # Count the number of pixels in the tumor region
    tumor_pixel_count = np.sum(tumor_mask)

    # Convert pixel count into physical units
    pixel_spacing = 1.227  # Adjusted this to the appropriate value according to inference derived from ct scan pixels vs average human lungs
    tumor_area_mm2 = tumor_pixel_count * pixel_spacing**2

    return tumor_area_mm2


if __name__ == '__main__':
    img_path = 'lungs.png'
    result = calculate_tumor_area(img_path)
    print("Segmented Tumor Area in physical units: "+ result +" mm^2")
