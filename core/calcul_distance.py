import numpy as np

class CalculDistance:
    """
    Classe permettant de calculer la distance entre deux vecteurs
    """
    
    def __init__(self, vector1=np.array([]), vector2=np.array([]), p=1.5):
        """
            Constructeur de la classe CalculDistance
            Usage:
                CalculDistance(vector1, vector2, p)
            Arguments:
                vector1: np.array - Vecteur 1, par défaut vide
                vector2: np.array - Vecteur 2, par défaut vide
                p: float - Paramètre de la distance de Minowski
        """
        self.vector1 = vector1
        self.vector2 = vector2
        self.p = p
        
    def __str__(self):
        return f"Vector 1: {self.vector1}\nVector 2: {self.vector2}"
    
    def __repr__(self):
        self.__str__()
    
    # Distance de Manhattan
    def manhattan(self):
        """
            Calcule la distance de Manhattan entre deux vecteurs
            Usage:
                distance = CalculDistance(vector1, vector2)
                distance.manhattan()
            Returns:
                float - Distance de Manhattan
        """
        return np.sum(np.abs(self.vector1 - self.vector2))
    
    # Distance euclidienne
    def euclidean(self):
        """
            Calcule la distance euclidienne entre deux vecteurs
            Usage:
                distance = CalculDistance(vector1, vector2)
                distance.euclidean()
            Returns:
                float - Distance euclidienne
        """
        return np.sqrt(np.sum((self.vector1 - self.vector2) ** 2))
    
    # Distance de Chebyshev
    def chebyshev(self):
        """
            Calcule la distance de Chebyshev entre deux vecteurs
            Usage:
                distance = CalculDistance(vector1, vector2)
                distance.chebyshev()
            Returns:
                float - Distance de Chebyshev
        """
        return np.max(np.abs(self.vector1 - self.vector2))
    
    # Intersection d'histogrammes
    def intersection(self):
        """
            Calcule l'intersection d'histogrammes entre deux vecteurs
            Usage:
                distance = CalculDistance(vector1, vector2)
                distance.intersection()
            Returns:
                float - Intersection d'histogrammes
        """
        return np.sum(np.minimum(self.vector1, self.vector2)) / (np.sum(self.vector2) + 1e-10)
    
    # Khi-2
    def khi2(self):
        """
            Calcule la distance Khi-2 entre deux vecteurs
            Usage:
                distance = CalculDistance(vector1, vector2)
                distance.khi2()
            Returns:
                float - Distance Khi-2
        """
        return np.sum((self.vector1 - self.vector2) ** 2 / (self.vector1 + self.vector2 + 1e-10)**2)
    
    # Distance de Minowski
    def minowski(self):
        """
            Calcule la distance de Minowski entre deux vecteurs
            Usage:
                distance = CalculDistance(vector1, vector2, p)
                distance.minowski()
            Returns:
                float - Distance de Minowski
        """
        return (np.sum(np.abs(self.vector1-self.vector2)**self.p ))**(1/self.p)