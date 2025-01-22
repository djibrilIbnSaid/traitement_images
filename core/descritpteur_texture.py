import numpy as np

class DescripteurTexture:
    
    @staticmethod
    def descripteur_texture(image, type, **kwargs):
        """
        Calcule le descripteur de texture d'une image.
        
        Args:
            image (np.array): Image en niveaux de gris à traiter.
            type (str): Type de descripteur de texture.
        
        Returns:
            np.array: Descripteur de texture.
        """
        if type == None:
            return np.array([])
        elif type == "LBP":
            return DescripteurTexture.lbp(image)
        elif type.lower() == "statistique":
            return DescripteurTexture.stats(image)
        elif type.lower() == "glcm":
            return DescripteurTexture.glcm(image, **kwargs)
        return np.array([])
    
    @staticmethod
    def lbp(image):
        """
        Calcule l'histogramme LBP d'une image.
        Mon ancienne fonction 'lbp_old' j'ai donné a chatgpt pour l'optimiser.
        
        Args:
            image (np.array): Image en niveaux de gris ou couleur à traiter.
        
        Returns:
            np.array: Histogramme LBP.
        """
        if image.ndim == 3:
            # Convertir en niveaux de gris si l'image est en couleur
            image = np.mean(image, axis=2).astype(np.uint8)
        
        rows, cols = image.shape
        tableau = np.zeros((rows - 2, cols - 2), dtype=np.uint8)

        # Créer un motif binaire basé sur les voisins avec un décalage pour chaque direction
        offsets = [
            (1, 1), (1, 0), (1, -1),
            (0, 1),         (0, -1),
            (-1, 1), (-1, 0), (-1, -1)
        ]
        
        # Calcul LBP vectorisé
        center = image[1:rows-1, 1:cols-1]
        for k, (di, dj) in enumerate(offsets):
            neighbor = image[1 + di:rows - 1 + di, 1 + dj:cols - 1 + dj]
            binary_pattern = (neighbor >= center).astype(np.uint8) << (7 - k)
            tableau |= binary_pattern

        # Calculer l'histogramme LBP
        hist, _ = np.histogram(tableau.ravel(), bins=256, range=(0, 256))
        return hist
    
    
    @staticmethod
    def stats(image):
        """
        Calcule les statistiques de base d'une image.
        
        Args:
            image (np.array): Image en niveaux de gris à traiter.
        
        Returns:
            np.array: Vecteur de statistiques.
        """
        return np.array([np.mean(image), np.std(image), np.percentile(image, 25), np.median(image), np.percentile(image, 75)])
    
    @staticmethod
    def lbp_old(image):
        """
        Calcule l'histogramme LBP d'une image.
        
        Args:
            image (np.array): Image en niveaux de gris à traiter.
        
        Returns:
            np.array: Histogramme LBP.
        """
        rows, cols = image.shape[0], image.shape[1]      
        tableau = np.zeros((rows-2, cols-2), dtype=np.uint8)
        
        # verifier si l'image est en niveaux de gris ou en couleur et faire la boucle en tenant compte de cela
        if len(image.shape) == 3:
            for c in range(3):
                sub_image = image[:, :, c]
                for i in range(1, rows - 1):
                    for j in range(1, cols - 1):
                        # Créer un motif binaire basé sur les voisins
                        binary_pattern = (
                            (sub_image[i+1, j+1] >= sub_image[i, j]) << 7 |
                            (sub_image[i+1, j] >= sub_image[i, j]) << 6 |
                            (sub_image[i+1, j-1] >= sub_image[i, j]) << 5 |
                            (sub_image[i, j+1] >= sub_image[i, j]) << 4 |
                            (sub_image[i, j-1] >= sub_image[i, j]) << 3 |
                            (sub_image[i-1, j+1] >= sub_image[i, j]) << 2 |
                            (sub_image[i-1, j] >= sub_image[i, j]) << 1 |
                            (sub_image[i-1, j-1] >= sub_image[i, j]) << 0
                        )
                        
                        # Mettre à jour le tableau
                        tableau[i-1, j-1] += binary_pattern
        else:
            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    # Créer un motif binaire basé sur les voisins
                    binary_pattern = (
                        (image[i+1, j+1] >= image[i, j]) << 7 |
                        (image[i+1, j] >= image[i, j]) << 6 |
                        (image[i+1, j-1] >= image[i, j]) << 5 |
                        (image[i, j+1] >= image[i, j]) << 4 |
                        (image[i, j-1] >= image[i, j]) << 3 |
                        (image[i-1, j+1] >= image[i, j]) << 2 |
                        (image[i-1, j] >= image[i, j]) << 1 |
                        (image[i-1, j-1] >= image[i, j]) << 0
                    )
                    
                    # Mettre à jour le tableau
                    tableau[i-1, j-1] += binary_pattern
        
        # Calculer l'histogramme LBP
        hist, _ = np.histogram(tableau.ravel(), bins=256, range=(0, 256))
        return hist
        
    
    @staticmethod
    def glcm(image, **kwargs):
        """
        Calcule la matrice de cooccurrence des niveaux de gris d'une image de manière optimisée.
        
        Args:
            image (np.array): Image en niveaux de gris à traiter.
        
        Returns:
            np.array: Matrice de cooccurrence des niveaux de gris.
        """
        # Paramètres
        distances = kwargs.get("distances", [1])
        angles = kwargs.get("angles", [0])
        levels = kwargs.get("levels", 256)
        
        # Matrice de cooccurrence
        glcm = np.zeros((levels, levels), dtype=np.int32)
        
        for angle in angles:
            for distance in distances:
                # Calcul des décalages
                dx = int(round(distance * np.cos(angle)))
                dy = int(round(distance * np.sin(angle)))
                
                # Pixels source et cible
                values_src = image[max(0, -dx):image.shape[0] - max(0, dx),
                                max(0, -dy):image.shape[1] - max(0, dy)]
                values_tgt = image[max(0, dx):image.shape[0] - max(0, -dx),
                                max(0, dy):image.shape[1] - max(0, -dy)]
                
                # Conversion explicite en indices entiers
                values_src = values_src.astype(int)
                values_tgt = values_tgt.astype(int)
                
                # Mise à jour de la matrice GLCM
                np.add.at(glcm, (values_src, values_tgt), 1)
        
        return glcm.ravel()
        