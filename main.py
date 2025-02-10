import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
import color_comp

# Load the image
# PLEASE BE IN THE CURRENT DIRECTORY, as in, inside fashionApp or else this WIILL throw error.
image_path = os.path.join(os.getcwd(), "images", "test6.jpg")
image = cv2.imread(image_path)

# Escape route if no image detected:
if image is None:
    print("Error: Image not found")
    exit()
    
# opencv makes the image in bgr format for god knows why so convert it to rgb
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load the pre-trained DNN model for face detection
model_file = "models/res10_300x300_ssd_iter_140000.caffemodel"
config_file = "models/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_file, model_file)

(height, width) = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# Perform face detection
net.setInput(blob)
detections = net.forward()

# Loop over the detections
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > 0.5:
        # Coordinates of the box surrounding the face
        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
        (x, y, x2, y2) = box.astype("int")
        face_region = image_rgb[y:y2, x:x2]

        pixels = face_region.reshape(-1, 3)

        # find the dominant color 
        kmeans = KMeans(n_clusters=1)
        kmeans.fit(pixels)
        dominant_color = kmeans.cluster_centers_.astype(int)[0]
        
        # Get the dominant color and convert to equivalent hex
        hex_color = "#{:02x}{:02x}{:02x}".format(dominant_color[0], dominant_color[1], dominant_color[2])

        # Print the hexadecimal color
        print(f"Detected Skin Tone: {hex_color}")
        
        # Ranking the complementary colors: 
        comp_palette = color_comp.get_comp_color(hex_color)
        print("Recommend Complementary Colors (Ranked): ")
        for rank, comp_color in enumerate(comp_palette, start =1):
            print(f" Rank {rank} : {comp_color}")

        # Draw rectangle around face (for visualisation) <---- serves no other purpose
        cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)

# Display the image with the detected face
cv2.imshow("Detected Face", image)
cv2.waitKey(0)
cv2.destroyAllWindows()