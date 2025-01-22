import os
import numpy as np
import pandas as pd
from PIL import Image
from core.conversion_espace_couleur import ConversionEspaceCouleur
from core.descripteur_couleurs import DescripteurCouleurs
from typing import List

def get_images(path):
    """
    Get all images from a directory and its subdirectories.

    Args:
        path (str): The path to the directory.
        
    Returns:
        list: A list of images.
    """
    import os
    images = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(root, file))
    return images


def str_to_vector(chaine):
    """
    Convert a string representation of a vector to a numpy array.
    
    Args:
        vector_str (str): The string representation of the vector.
        
    Returns:
        np.ndarray: The numpy array representation of the vector.
    """
    # vector_str = vector_str.strip('[]')
    # return np.fromstring(vector_str, sep=' ')
    return list(map(float, chaine.split(', ')))

def vector_to_str(vecteur):
    """
    Convert a numpy array to a string representation.
    
    Args:
        vecteur (np.ndarray): The numpy array.
        
    Returns:
        str: The string representation of the numpy array.
    """
    return ', '.join(map(str, vecteur))

# Sauvegarder la base de données en fonction de la conversion de l'espace de couleur avec l'histogramme niveau de gris
def save_database_gray_histogram(images, path, density=False):
    """
    Sauvegarde la base de données en fonction de la conversion de l'espace de couleur avec l'histogramme niveau de gris.

    Args:
        images (list): La liste d'images.
        path (str): Le fichier csv.
    """
    df = pd.read_csv(path)
    col_name = 'histogram_norm_gray' if density else 'histogram_gray'
    for image in images:
        image_name = os.path.basename(image)
        img = Image.open(image)
        img = np.array(img)
        img_gray = ConversionEspaceCouleur.rgb_to_gray(img)
        hist = DescripteurCouleurs.histogramme_rgb(img_gray, density=density)
        if df[df['image'] == image_name].empty:
            df = df.append({'image': image_name, col_name: hist}, ignore_index=True)
        else:
            df.loc[df['image'] == image_name, col_name] = hist
    df.to_csv(path, index=False)
    

def apply_functions_to_images(image_paths: List[str], function_names: List[str], output_csv: str):
    """
    Applique des fonctions de conversion à une ou plusieurs images et sauvegarde les résultats dans un fichier CSV.

    Args:
        image_paths (List[str]): Liste des chemins d'accès des images.
        function_names (List[str]): Liste des noms des fonctions à appliquer aux images.
        output_csv (str): Chemin du fichier CSV de sortie.

    Returns:
        None
    """
    # Obtenez les méthodes de la classe ConversionEspaceCouleur
    methods = {func: getattr(ConversionEspaceCouleur, func) for func in dir(ConversionEspaceCouleur) 
               if callable(getattr(ConversionEspaceCouleur, func)) and not func.startswith("__")}
    
    # Tri des noms de fonctions par ordre alphabétique et en minuscules
    function_names = sorted([name.lower() for name in function_names])
    
    # Créez une liste pour stocker les résultats
    results = []

    # Parcourez chaque image
    for image_path in image_paths:
        # Chargez l'image en tant que tableau NumPy
        
        image = np.array(Image.open(image_path))
        
        # Créez un dictionnaire pour stocker les résultats pour cette image
        result = {'image_path': image_path}
        
        # Appliquez chaque fonction spécifiée
        for func_name in function_names:
            func_name_camel = func_name.lower()  # Assurez-vous que les noms correspondent au format
            if func_name_camel in methods:
                # Exécutez la fonction
                output = methods[func_name_camel](image)
                # Sauvegarder un résumé (par ex., moyenne des valeurs de sortie) dans le CSV
                result[func_name] = np.mean(output)
            else:
                result[func_name] = None
        
        # Ajoutez le résultat de cette image à la liste
        results.append(result)
    
    # Créez un DataFrame pandas à partir des résultats
    df = pd.DataFrame(results)
    
    # Sauvegardez le DataFrame dans un fichier CSV
    df.to_csv(output_csv, index=False)


def convert_and_resize_images(src_folder, dest_folder, size=(448, 448)):
    """
    Cette fonction parcourt les dossiers et sous-dossiers du dossier source,
    crée les mêmes sous-dossiers dans le dossier de destination, redimensionne
    chaque image à la taille donnée, et les enregistre sous le même nom.
    
    :param src_folder: Chemin du dossier source
    :param dest_folder: Chemin du dossier de destination
    :param size: Taille des images redimensionnées (largeur, hauteur)
    """
    
    # Parcourt le dossier source
    for root, dirs, files in os.walk(src_folder):
        # Génère le chemin du dossier correspondant dans le dossier de destination
        relative_path = os.path.relpath(root, src_folder)
        dest_path = os.path.join(dest_folder, relative_path)
        
        # Crée le dossier dans le dossier de destination s'il n'existe pas
        os.makedirs(dest_path, exist_ok=True)
        
        for file in files:
            # Ignore les fichiers .DS_Store
            if file == '.DS_Store':
                continue
            
            # Chemin complet de l'image dans le dossier source
            src_file_path = os.path.join(root, file)
            # Chemin complet pour enregistrer l'image redimensionnée
            dest_file_path = os.path.join(dest_path, file)
            
            try:
                # Ouvrir l'image et la redimensionner
                with Image.open(src_file_path) as img:
                    img_resized = img.resize(size, Image.LANCZOS)
                    # Sauvegarde l'image redimensionnée dans le dossier de destination
                    img_resized.save(dest_file_path)
                print(f"Image '{file}' convertie et enregistrée dans '{dest_file_path}'")
            except Exception as e:
                print(f"Erreur avec l'image {file} : {e}")