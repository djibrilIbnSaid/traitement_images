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


def convert_to_dataframe(data: List, columns=[]) -> pd.DataFrame:
    """
    Convert a list of dictionaries to a pandas DataFrame.
    
    Args:
        data (list): A list of dictionaries.
        
    Returns:
        pd.DataFrame: The pandas DataFrame.
    """
    df = pd.DataFrame(data, columns=columns)
    # df['image'] = df['image'].apply(lambda x: x.split('_')[0])
    # pour la precision renvoyer 3 chiffres après la virgule
    df['precision'] = df['precision'].apply(lambda x: round(float(x), 2))
    return df


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