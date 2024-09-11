""" import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os
import sys

# Ajouter le répertoire 'scripts' au chemin de recherche des modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

# Charger les données à partir du fichier CSV
data_file = os.path.join('..', 'data', 'sample_data.csv')
data = pd.read_csv(data_file)

# Extraire les points et les distances
points = data[['x', 'y']].values
distances = data['distance'].values

# Créer une grille 2D
grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]

# Interpoler les données
from interpolation import interpolate_data  # Importer directement
grid_z = interpolate_data(points, distances, grid_x, grid_y)

# Visualiser et sauvegarder l'image
plt.imshow(grid_z.T, extent=(0,1,0,1), origin='lower')
plt.colorbar()
output_file = os.path.join('..', 'output', 'ultrasound_image.png')
plt.savefig(output_file)
plt.show()
 """
""" 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os
import sys

# Ajouter le répertoire 'scripts' au chemin de recherche des modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

# Charger les données à partir du fichier CSV
data_file = os.path.join('..', 'data', 'sample_data.csv')
data = pd.read_csv(data_file)

# Extraire les points et les distances
points = data[['x', 'y']].values
distances = data['distance'].values

# Créer une grille 2D
grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]

# Interpoler les données
from interpolation import interpolate_data  # Importer directement
grid_z = interpolate_data(points, distances, grid_x, grid_y)

# Ajouter du bruit à l'image pour simuler l'aspect échographique
noise = np.random.normal(0, 0.1, grid_z.shape)
grid_z_noisy = grid_z + noise

# Appliquer l'égalisation d'histogramme pour améliorer le contraste
from skimage import exposure
grid_z_adjusted = exposure.equalize_hist(grid_z_noisy)

# Visualiser et sauvegarder l'image avec bruit et ajustement de contraste
plt.imshow(grid_z_adjusted.T, extent=(0,1,0,1), origin='lower', cmap='gray')
plt.colorbar()
output_file = os.path.join('..', 'output', 'ultrasound_image_adjusted.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.show()
 """
""" 
# generate_image.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageEnhance
from interpolation import interpolate_data  # Assurez-vous que le chemin est correct

def enhance_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    enhanced = cv2.equalizeHist(blurred)
    return enhanced

def enhance_image_pillow(image_path):
    with Image.open(image_path) as img:
        enhancer = ImageEnhance.Contrast(img)
        img_enhanced = enhancer.enhance(2.0)  # Augmenter le contraste
        img_enhanced.save('../output/enhanced_ultrasound_image_pillow.png')

# Charger les données
data_file = '../data/sample_data.csv'  # Chemin relatif basé sur la structure
data = pd.read_csv(data_file)
points = data[['x', 'y']].values
distances = data['distance'].values

# Créer une grille 2D
grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]

# Interpoler les données
grid_z = interpolate_data(points, distances, grid_x, grid_y)

# Visualiser et sauvegarder l'image
plt.imshow(grid_z.T, extent=(0, 1, 0, 1), origin='lower', cmap='gray')
plt.colorbar()
plt.title('Image Echographique')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('../output/ultrasound_image.png')

# Post-traitement avec OpenCV
image = cv2.imread('../output/ultrasound_image.png')
enhanced_image = enhance_image(image)
cv2.imwrite('../output/enhanced_ultrasound_image.png', enhanced_image)

# Post-traitement avec Pillow
enhance_image_pillow('../output/ultrasound_image.png')
 """
""" 
import csv
import numpy as np
import matplotlib.pyplot as plt

def generate_image_from_csv(csv_file):
    x_vals = []
    y_vals = []
    distances = []

    # Lire les données depuis le fichier CSV
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x_vals.append(int(row['x']))
            y_vals.append(int(row['y']))
            distances.append(float(row['distance']))

    # Définir la taille de la grille
    grid_size = max(max(x_vals), max(y_vals)) + 1
    distance_matrix = np.zeros((grid_size, grid_size))

    # Remplir la matrice avec les distances
    for x, y, dist in zip(x_vals, y_vals, distances):
        distance_matrix[y, x] = dist

    # Générer une image avec Matplotlib
    plt.imshow(distance_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Distance (cm)')
    plt.title('Représentation des distances en image')
    plt.show()

if __name__ == '__main__':
    # Assurez-vous que le chemin vers le fichier CSV est correct
    generate_image_from_csv('../data/simulated_data.csv')
 """


""" import csv
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# Fonction pour générer une image à partir du fichier CSV
def generate_ultrasound_image_from_csv(csv_file_path, scale_factor=80):
    # Initialisation des listes pour les coordonnées x, y et les distances
    x_vals = []
    y_vals = []
    distances = []

    # Lire les données du fichier CSV
    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                x = int(row['x'])
                y = int(row['y'])
                distance = float(row['distance'])
                x_vals.append(x)
                y_vals.append(y)
                distances.append(distance)
            except ValueError as e:
                print(f"Erreur de conversion : {e}, ligne: {row}")

    # Vérification si les listes ne sont pas vides
    if not x_vals or not y_vals:
        print("Erreur : Les listes x_vals ou y_vals sont vides.")
        return

    # Taille de la grille basée sur les valeurs maximales de x et y
    grid_size = max(max(x_vals), max(y_vals)) + 1

    # Générer une image avec des valeurs en niveaux de gris
    image_array = np.zeros((grid_size, grid_size))

    for x, y, distance in zip(x_vals, y_vals, distances):
        # Convertir la distance en intensité de gris (0 = noir, 255 = blanc)
        intensity = np.interp(distance, (min(distances), max(distances)), (255, 0))  # Inversé pour que 255 soit le plus proche
        image_array[y, x] = intensity

    # Appliquer un filtre gaussien pour lisser l'image
    image_array = gaussian_filter(image_array, sigma=0.2)  # Un sigma plus élevé pour un meilleur lissage

    # Ajouter du bruit réaliste pour simuler une échographie
    image_array = add_structured_noise(image_array, noise_factor=0.05 )

    # Créer une image avec PIL en niveaux de gris
    image = Image.fromarray(image_array.astype(np.uint8), mode='L')

    # Redimensionner l'image selon le facteur de mise à l'échelle (avec interpolation bilinéaire pour lisser)
    width, height = image.size
    new_size = (width * scale_factor, height * scale_factor)
    image = image.resize(new_size, Image.BICUBIC)  # Utilisation de BICUBIC pour un lissage plus fin

    # Sauvegarder l'image résultante
    image.save('../output/ultrasound_image_realistic.png')
    print(f"Image générée et sauvegardée sous 'ultrasound_image_realistic.png' avec une taille de {new_size}.")

# Fonction pour ajouter un bruit structuré réaliste à l'image
def add_structured_noise(image_array, noise_factor=2):
    # Créer un bruit gaussien structuré pour simuler un effet de grain dans l'image d'échographie
    noise = np.random.randn(*image_array.shape) * 255 * noise_factor
    noisy_image = image_array + noise
    
    # Limiter les valeurs entre 0 et 255
    noisy_image = np.clip(noisy_image, 0, 255)
    
    # Appliquer un filtre gaussien léger pour lisser encore un peu le bruit
    noisy_image = gaussian_filter(noisy_image, sigma=0.01)
    
    return noisy_image

# Appeler la fonction pour générer l'image à partir du fichier CSV
if __name__ == '__main__':
    generate_ultrasound_image_from_csv('../data/simulated_data.csv', scale_factor=80)


 """

""" 
import csv
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# Fonction pour générer une image à partir du fichier CSV
def generate_ultrasound_image_from_csv(csv_file_path, scale_factor=100):
    # Initialisation des listes pour les coordonnées x, y et les distances
    x_vals = []
    y_vals = []
    distances = []

    # Lire les données du fichier CSV
    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                x = int(row['x'])
                y = int(row['y'])
                distance = float(row['distance'])
                x_vals.append(x)
                y_vals.append(y)
                distances.append(distance)
            except ValueError as e:
                print(f"Erreur de conversion : {e}, ligne: {row}")

    # Vérification si les listes ne sont pas vides
    if not x_vals or not y_vals:
        print("Erreur : Les listes x_vals ou y_vals sont vides.")
        return

    # Taille de la grille basée sur les valeurs maximales de x et y
    grid_size = max(max(x_vals), max(y_vals)) + 1

    # Générer une image avec des valeurs en niveaux de gris
    image_array = np.zeros((grid_size, grid_size))

    for x, y, distance in zip(x_vals, y_vals, distances):
        # Convertir la distance en intensité de gris (0 = noir, 255 = blanc)
        intensity = np.interp(distance, (min(distances), max(distances)), (255, 0))  # Inversé pour que 255 soit le plus proche
        image_array[y, x] = intensity

    # Appliquer un filtre gaussien pour lisser l'image
    image_array = gaussian_filter(image_array, sigma=2)  # Un sigma plus élevé pour un meilleur lissage

    # Ajouter du bruit réaliste pour simuler une échographie
    image_array = add_structured_noise(image_array, noise_factor=0.05)

    # Créer une image avec PIL en niveaux de gris
    image = Image.fromarray(image_array.astype(np.uint8), mode='L')

    # Redimensionner l'image selon le facteur de mise à l'échelle (avec interpolation bicubique pour lisser)
    width, height = image.size
    new_size = (width * scale_factor, height * scale_factor)  # Taille plus grande
    image = image.resize(new_size, Image.BICUBIC)  # Utilisation de BICUBIC pour un lissage plus fin

    # Sauvegarder l'image résultante
    image.save('../output/ultrasound_image_large.png')
    print(f"Image générée et sauvegardée sous 'ultrasound_image_large.png' avec une taille de {new_size}.")

# Fonction pour ajouter un bruit structuré réaliste à l'image
def add_structured_noise(image_array, noise_factor=0.05):
    # Créer un bruit gaussien structuré pour simuler un effet de grain dans l'image d'échographie
    noise = np.random.randn(*image_array.shape) * 255 * noise_factor
    noisy_image = image_array + noise
    
    # Limiter les valeurs entre 0 et 255
    noisy_image = np.clip(noisy_image, 0, 255)
    
    # Appliquer un filtre gaussien léger pour lisser encore un peu le bruit
    noisy_image = gaussian_filter(noisy_image, sigma=0.5)
    
    return noisy_image

# Appeler la fonction pour générer l'image à partir du fichier CSV
if __name__ == '__main__':
    generate_ultrasound_image_from_csv('../data/simulated_data.csv', scale_factor=100)
 """


""" 
import csv
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# Fonction pour générer une image à partir du fichier CSV
def generate_ultrasound_image_from_csv(csv_file_path, target_size=(568, 470)):
    # Initialisation des listes pour les coordonnées x, y et les distances
    x_vals = []
    y_vals = []
    distances = []

    # Lire les données du fichier CSV
    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                x = int(row['x'])
                y = int(row['y'])
                distance = float(row['distance'])
                x_vals.append(x)
                y_vals.append(y)
                distances.append(distance)
            except ValueError as e:
                print(f"Erreur de conversion : {e}, ligne: {row}")

    # Vérification si les listes ne sont pas vides
    if not x_vals or not y_vals:
        print("Erreur : Les listes x_vals ou y_vals sont vides.")
        return

    # Taille de la grille basée sur les valeurs maximales de x et y
    grid_size = max(max(x_vals), max(y_vals)) + 1

    # Générer une image avec des valeurs en niveaux de gris
    image_array = np.zeros((grid_size, grid_size))

    for x, y, distance in zip(x_vals, y_vals, distances):
        # Convertir la distance en intensité de gris (0 = noir, 255 = blanc)
        intensity = np.interp(distance, (min(distances), max(distances)), (255, 0))  # Inversé pour que 255 soit le plus proche
        image_array[y, x] = intensity

    # Appliquer un filtre gaussien pour lisser l'image
    image_array = gaussian_filter(image_array, sigma=2)  # Lissage avec sigma

    # Ajouter du bruit réaliste pour simuler une échographie
    image_array = add_structured_noise(image_array, noise_factor=0.05)

    # Créer une image avec PIL en niveaux de gris
    image = Image.fromarray(image_array.astype(np.uint8), mode='L')

    # Redimensionner l'image à la taille exacte (568x470)
    image = image.resize(target_size, Image.BICUBIC)  # Utilisation de BICUBIC pour un lissage fin

    # Sauvegarder l'image résultante
    image.save('../output/ultrasound_image_568x470.png')
    print(f"Image générée et sauvegardée sous 'ultrasound_image_568x470.png' avec une taille de {target_size}.")

# Fonction pour ajouter un bruit structuré réaliste à l'image
def add_structured_noise(image_array, noise_factor=0.05):
    # Créer un bruit gaussien structuré pour simuler un effet de grain dans l'image d'échographie
    noise = np.random.randn(*image_array.shape) * 255 * noise_factor
    noisy_image = image_array + noise
    
    # Limiter les valeurs entre 0 et 255
    noisy_image = np.clip(noisy_image, 0, 255)
    
    # Appliquer un filtre gaussien léger pour lisser encore un peu le bruit
    noisy_image = gaussian_filter(noisy_image, sigma=0.5)
    
    return noisy_image

# Appeler la fonction pour générer l'image à partir du fichier CSV
if __name__ == '__main__':
    generate_ultrasound_image_from_csv('../data/simulated_data.csv', target_size=(568, 470))
 """







import numpy as np
import csv
from PIL import Image
from scipy.ndimage import gaussian_filter
import cv2

# Fonction pour ajouter du bruit speckle
def add_speckle_noise(image_array, noise_factor=0.05):
    noise = noise_factor * np.random.randn(*image_array.shape)
    image_array += image_array * noise
    image_array = np.clip(image_array, 0, 255)  # Garder les valeurs entre 0 et 255
    return image_array

# Fonction pour appliquer un filtre passe-haut
def apply_high_pass_filter(image_array, sigma=2):
    blurred = gaussian_filter(image_array, sigma=sigma)
    high_pass = image_array - blurred
    high_pass = np.clip(high_pass, 0, 255)  # Assurer les valeurs entre 0 et 255
    return high_pass

# Fonction pour appliquer un filtre anisotrope
def apply_anisotropic_filter(image_array):
    # Vérifier si l'image est en niveaux de gris ou en couleur
    if len(image_array.shape) == 2:  # Image en niveaux de gris
        image_cv = (image_array / 255.0).astype(np.float32)
        # Convertir l'image en niveaux de gris à une image BGR
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_GRAY2BGR)
    elif len(image_array.shape) == 3 and image_array.shape[2] == 3:  # Image en couleur
        image_cv = (image_array / 255.0).astype(np.float32)
    else:
        raise ValueError("L'image d'entrée doit être en niveaux de gris ou une image couleur à 3 canaux.")

    # Assurez-vous que l'image est de type uint8
    if image_cv.dtype != np.uint8:
        image_cv = np.uint8(image_cv * 255)

    # Appliquer le filtre de diffusion anisotrope
    filtered_image = cv2.ximgproc.anisotropicDiffusion(image_cv, alpha=0.15, K=25, niters=20)
    filtered_image = (filtered_image * 255).astype(np.uint8)
    
    # Convertir l'image filtrée en niveaux de gris si nécessaire
    if len(filtered_image.shape) == 3 and filtered_image.shape[2] == 3:
        filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

    return filtered_image

# Fonction pour améliorer le contraste
def enhance_contrast(image_array):
    image_array = cv2.normalize(image_array, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return image_array

# Fonction principale pour générer l'image échographique à partir du CSV
def generate_ultrasound_image_from_csv(csv_file_path, target_size=(568, 470)):
    x_vals = []
    y_vals = []
    distances = []

    # Lire les données du fichier CSV
    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                x = int(row['x'])
                y = int(row['y'])
                distance = float(row['distance'])
                x_vals.append(x)
                y_vals.append(y)
                distances.append(distance)
            except ValueError as e:
                print(f"Erreur de conversion : {e}, ligne: {row}")

    if not x_vals or not y_vals:
        print("Erreur : Les listes x_vals ou y_vals sont vides.")
        return

    grid_size = max(max(x_vals), max(y_vals)) + 1
    image_array = np.zeros((grid_size, grid_size))

    # Remplir la matrice d'image avec les distances
    for x, y, distance in zip(x_vals, y_vals, distances):
        intensity = np.interp(distance, (min(distances), max(distances)), (255, 0))  # Inversé pour que 255 soit le plus proche
        image_array[y, x] = intensity

    # Ajouter un bruit speckle
    image_array = add_speckle_noise(image_array, noise_factor=0.04)  # Réduire le bruit speckle

    # Appliquer un filtre passe-haut
    image_array = apply_high_pass_filter(image_array)

    # Appliquer un filtre anisotrope
    image_array = apply_anisotropic_filter(image_array)

    # Amélioration du contraste
    image_array = enhance_contrast(image_array)

    # Créer une image en niveaux de gris
    image = Image.fromarray(image_array.astype(np.uint8), mode='L')

    # Redimensionner à la taille souhaitée (568x470)
    image = image.resize(target_size, Image.BICUBIC)

    # Sauvegarder l'image
    image.save('../output/ultrasound_realistic_image.png')
    print(f"Image générée et sauvegardée sous 'ultrasound_realistic_image.png' avec une taille de {target_size}.")

# Exécuter la génération d'image avec un fichier CSV
generate_ultrasound_image_from_csv('../data/simulated_data.csv', target_size=(568, 470))
