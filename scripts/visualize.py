import cv2
import numpy as np
import os

# Charger l'image générée
image_file = os.path.join('output', 'ultrasound_image.png')
image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

# Appliquer un filtre de flou gaussien
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Détection des contours
edges = cv2.Canny(blurred, 100, 200)

# Afficher l'image floutée et les contours
cv2.imshow('Blurred Image', blurred)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
