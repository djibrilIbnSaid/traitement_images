import numpy as np

class DescripteurCouleurs:
    
    @staticmethod
    def blob(image, espace_color, interval=3, dim_fen=4, canaux_rgb_indexe=(2, 2, 2)):
        """
        Calcule l'histogramme de blobs d'une image.
        
        Args:
            image (np.array): Image à traiter.
            type (str): Type de l'histogramme.
            interval (int): Intervalle.
            dim_fen (int): Dimension de la fenêtre.
            
        Returns:
            np.array: Histogramme de blobs.
        """
        nb_lignes = 0 # Nombre de lignes en fonction du contenu de l'image
        if espace_color == "gray basic" or espace_color == "gray 709" or espace_color == "gray 601":
            nb_lignes = 256
        elif espace_color == "hsv indexe" or espace_color == "hsl indexe":
            nb_lignes = 360
            # faites modification pour hsv et hsl et retourner l'histogramme
        elif espace_color == "rgb indexe" and canaux_rgb_indexe is not None:
            nb_lignes = canaux_rgb_indexe[0] * canaux_rgb_indexe[1] * canaux_rgb_indexe[2]
        else:
            return image
            
        
        rows, cols = image.shape
        tableau = np.zeros((nb_lignes, interval), dtype=int)
        
        # Précalculer le nombre d'éléments par sous-matrice
        total_elements = dim_fen * dim_fen
        
        # Parcours des sous-matrices avec un stride de 1
        for i in range(rows - dim_fen + 1):
            for j in range(cols - dim_fen + 1):
                # Extraction directe de la sous-matrice
                subimage = image[i:i + dim_fen, j:j + dim_fen].flatten()
                
                # Comptage des éléments uniques et leurs fréquences
                unique_elements, counts = np.unique(subimage, return_counts=True)
                
                # Mettre à jour le tableau en fonction des fréquences
                for element, count in zip(unique_elements, counts):
                    element = int(element)
                    if element >= nb_lignes:  # Ignorer si élément hors des limites de nb_lignes
                        continue
                    frequence = count / total_elements
                    col = int(frequence * interval)
                    col = min(col, interval - 1)
                    tableau[element, col] += 1
        
        return tableau    
        