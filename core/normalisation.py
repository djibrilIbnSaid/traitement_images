import numpy as np


class Normalisation:
    
    @staticmethod
    def histogramme(type, image):
        """
        Calcule l'histogramme d'une image.
        
        Args:
            type (str): Type normalisation de l'histogramme.
            image (np.array): Image à traiter.
        
        Returns:
            np.array: Histogramme.
        """
        if type == "Occurence" or type == "Frequence":
            return Normalisation.histogramme_image(image, density=type == "Frequence")
        elif type == "Statistique":
            return Normalisation.statistiques(image)
        elif type == "MinMax":
            return Normalisation.min_max(image)
        elif type == "Rang":
            return Normalisation.rang(image)
        return np.array([])
    
    @staticmethod
    def histogramme_image(image, density):
        """
        Calcule l'histogramme RGB d'une image.
        
        Args:
            image (np.array): Image à traiter.
        
        Returns:
            np.array: Histogramme RGB.
        """
        if len(image.shape) == 2:
            hist = np.histogram(image.flatten(), bins=np.arange(256), density=density)
            return hist[0]
        hist_r = np.histogram(image[:, :, 0].flatten(), bins=np.arange(256), density=density)
        hist_g = np.histogram(image[:, :, 1].flatten(), bins=np.arange(256), density=density)
        hist_b = np.histogram(image[:, :, 2].flatten(), bins=np.arange(256), density=density)
        return np.concatenate((hist_r[0], hist_g[0], hist_b[0]))
    
    @staticmethod
    def statistiques(image):
        """
        Calcule les statistiques d'une image.
        
        Args:
            image (np.array): Image à traiter.
        
        Returns:
            np.array: Statistiques d'image.
        """
        return np.array([np.mean(image), np.std(image), np.percentile(image, 25), np.median(image), np.percentile(image, 75)])
    
    
    @staticmethod
    def min_max(image):
        """
        Calcule la normalisation min-max.
        
        Args:
            None
        
        Returns:
            None
        """
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        return image.flatten()
    
    @staticmethod
    def rang(image):
        """
        Calcule la normalisation par le rang.
        
        Args:
            image (np.ndarray): L'image à normaliser.
        
        Returns:
            np.ndarray: L'image normalisée par le rang.
        """
        rang_image = np.linalg.matrix_rank(image)
        normalized_image = np.zeros_like(image, dtype=float)
        
        for i in range(image.shape[2]):  # Pour chaque canal de couleur
            normalized_image[:, :, i] = image[:, :, i] / rang_image
        
        return normalized_image.reshape(-1)
