import numpy as np
from PIL import Image
from core.descripteur_cnn import DescripteurCNN
from core.descritpteur_texture import DescripteurTexture
from core.discripteur_forme import DescripteurForme
from core.utils import get_images, str_to_vector, vector_to_str, convert_to_dataframe
from core.calcul_distance import CalculDistance
from core.descripteur_couleurs import DescripteurCouleurs
from core.normalisation import Normalisation
from core.conversion_espace_couleur import *
from db.sqlite_db import SqliteDB

class Traitement:
    """
        Classe permettant de rechercher des images par similarité
    """
    
    def __init__(self, path_images='BD_images_resized', db_connect='db.sqlite3', table_name='images_vectors'):
        """
            Constructeur de la classe Traitement
            Usage:
                Traitement(path_images)
            Arguments:
                path_images: str - Chemin des images
            Returns:
                None
                
        """
        self.path_images = path_images
        self.db_connect = SqliteDB(db_connect, offline=True)
        self.table_name = table_name
        self.db_connect.initial_db(self.table_name)
        self.precisions = []
        self.get_mean_average_precision(limit=10)
        self.cnn = DescripteurCNN()
        
        
    def __str__(self):
        return f"Chemin des images: {self.path_images}"
    
    def __repr__(self):
        self.__str__()
    
    def recherche_images(self, base_image=None, distance=None, color_descriptor=None, espace_color=None, nomalisation=None, shape_descriptor=None, filter=None, texture_descriptor=None, cnn_descriptor=None, nb_responses=5, p_minowski=1.5, canal_r=2, canal_g=2, canal_b=2, dim_fen=3, interval=4, offline=True, maen_average_precision=False):
        """
            Recherche d'images par similarité
            Usage:
                traitement = Traitement(base_image, distance, espace_color, nomalisation, shape_descriptor, filter texture_descriptor, cnn_descriptor, nb_responses, nb_lines, nb_columns)
                traitement.recherche_images()
            Returns:
                list - Galerie d'images
        """
        # print(f"""distance: {distance} | color_descriptor: {color_descriptor} | espace_color: {espace_color} | nomalisation: {nomalisation}
        #       | shape_descriptor: {shape_descriptor} | filter: {filter} | texture_descriptor: {texture_descriptor} | cnn_descriptor: {cnn_descriptor}
        #       | nb_responses: {nb_responses} | p_minowski: {p_minowski} | canal_r: {canal_r} | canal_g: {canal_g} | canal_b: {canal_b}
        #       | dim_fen: {dim_fen} | interval: {interval} | offline: {offline} | maen_average_precision: {maen_average_precision}""")
        
        if base_image is None:
            return[]
        cnn = None
        if cnn_descriptor is not None:
            cnn = self.cnn
        
        id = f'_{color_descriptor}_{espace_color}_{nomalisation}_{shape_descriptor}_{filter}_{texture_descriptor}_{cnn_descriptor}_{canal_r}_{canal_g}_{canal_b}_{dim_fen}_{interval}'

        if base_image is None:
            return []

        self.db_connect.offline = offline
        name_base = base_image.split('/')[-1]
        base_image = Image.open(base_image).convert("RGB").resize((448, 448)) # resize image
        
        calcul_distance = CalculDistance()
        db_images = get_images(self.path_images)
        images = []
        
        key = f"{name_base}{id}"
        if self.db_connect.check_exist(key, table_name=self.table_name):
            vector_base = self.db_connect.select(self.table_name, where="name = ?", params=(key,))
            vector_base = str_to_vector(vector_base[0][1])
            calcul_distance.vector1 = np.array(vector_base)
        else:
            base_image = base_image.convert("RGB")
            base_image = np.array(base_image)
            
            vecteur_espace_couleur_base, image = self.change_color_space(base_image, espace_color, nomalisation, color_descriptor)
            vecteur_texture_base = DescripteurTexture.descripteur_texture(base_image, texture_descriptor)
        
            calcul_distance.vector1 = np.concatenate((vecteur_espace_couleur_base, vecteur_texture_base))
            calcul_distance.vector1 = np.concatenate((calcul_distance.vector1, self.use_cnn_descriptor(base_image, cnn_descriptor, cnn_model=cnn)))
            
            calcul_distance.vector1 = np.concatenate((calcul_distance.vector1, DescripteurForme.descripteur_forme(base_image, shape_descriptor, filter)))
            
            # Enregistrement du vecteur dans la base de données
            self.db_connect.insert(self.table_name, [key, vector_to_str(calcul_distance.vector1)])
        
        for filename in db_images:
            key_2 = f"{filename.split('/')[-1]}{id}"
            if self.db_connect.check_exist(key_2):
                vector_base2 = self.db_connect.select(self.table_name, where="name = ?", params=(key_2,))
                calcul_distance.vector2 = str_to_vector(vector_base2[0][1]) 
                images.append({'image': Image.open(filename).convert("RGB"), 'distance': self.cal_distance(distance, calcul_distance), 'name': filename.split('/')[-1]})
                continue
            
            image_o = Image.open(filename).convert("RGB")
            image = np.array(image_o)

            vecteur_espace_couleur, _ = self.change_color_space(image, espace_color, nomalisation, color_descriptor)
            vecteur_texture = DescripteurTexture.descripteur_texture(image, texture_descriptor)
            
            calcul_distance.vector2 = np.concatenate((vecteur_espace_couleur, vecteur_texture))
            calcul_distance.vector2 = np.concatenate((calcul_distance.vector2, self.use_cnn_descriptor(image, cnn_descriptor, cnn_model=cnn)))
            
            calcul_distance.vector2 = np.concatenate((calcul_distance.vector2, DescripteurForme.descripteur_forme(image, shape_descriptor, filter)))
            
            # Enregistrement du vecteur dans la base de données
            self.db_connect.insert(self.table_name, [key_2, vector_to_str(calcul_distance.vector2)])

            images.append({'image': image_o, 'distance': self.cal_distance(distance, calcul_distance), 'name': filename.split('/')[-1]})
            
            
        # images = sorted(images, key=lambda x: x['distance'])
        images = sorted(images, key=lambda x: (x["distance"], x["name"].startswith(name_base[:7])), reverse=False)
        
        if maen_average_precision:
            return self.mean_average_precision(name_base, images), f"{color_descriptor}_{espace_color}_{nomalisation}_{shape_descriptor}_{filter}_{texture_descriptor}_{cnn_descriptor}_{canal_r}_{canal_g}_{canal_b}_{dim_fen}_{interval}_{distance}"
        
        
        images = images[:nb_responses]
        return [image['image'] for image in images]
        
    
    

    def change_color_space(self, image, color_space, nomalisation, color_descriptor, cr=2, cg=2, cb=2, interval=3, dim_fen=4):
        """
            Convertir l'espace de couleur d'une image
            Usage:
                change_color_space(image, color_space, nomalisation, color_descriptor, cr, cg, cb, interval, dim_fen)
            Returns:
                    np.array - Image convertie
        """
        if color_descriptor is None:
            # return Normalisation.histogramme(nomalisation, image), image
            return np.array([]), image
        if color_descriptor == "Blobs":
            canaux_rgb_indexe = (cr, cg, cb)
            image = self.use_color_descriptor(image, color_descriptor, color_space, canaux_rgb_indexe, interval, dim_fen)
            return Normalisation.histogramme(nomalisation, image), image
            
        
        if color_space == "rgb":
            image = ConversionEspaceCouleur.rgb(image)
        elif color_space == "rgb normalized":
            image = ConversionEspaceCouleur.rgb_normalized(image)
        elif color_space == "gray basic":
            image = ConversionEspaceCouleur.rgb_to_gray_basic(image)
        elif color_space == "gray 709":
            image = ConversionEspaceCouleur.rgb_to_gray_709(image)
        elif color_space == "gray 601":
            image = ConversionEspaceCouleur.rgb_to_gray_601(image)
        elif color_space == "yiq":
            image = ConversionEspaceCouleur.rgb_to_yiq(image)
        elif color_space == "yuv":
            image = ConversionEspaceCouleur.rgb_to_yuv(image)
        elif color_space == "l1l2l3":
            image = ConversionEspaceCouleur.rgb_to_l1l2l3(image)
        elif color_space == "norm rgb":
            image = ConversionEspaceCouleur.rgb_to_normalized_rgb(image)
        elif color_space == "hsv":
            image = ConversionEspaceCouleur.rgb_to_hsv(image)
        elif color_space == "hsl":
            image = ConversionEspaceCouleur.rgb_to_hsl(image)
        elif color_space == "lab":
            image = ConversionEspaceCouleur.rgb_to_lab(image)
        elif color_space == "luv":
            image = ConversionEspaceCouleur.rgb_to_luv(image)
        elif color_space == "cmyk":
            image = ConversionEspaceCouleur.rgb_to_cmyk(image)
        elif color_space == "rgb indexe":
            image = ConversionEspaceCouleur.rgb_indexer(image)
            
            
            
        
        

        return Normalisation.histogramme(nomalisation, image), image
    
        
    def cal_distance(self, distance, calcul_distance):
        """
            Calcul de la distance entre deux vecteurs
            Usage:
                cal_distance(distance, calcul_distance)
            Returns:
                float - Distance entre deux vecteurs
        """
        if distance == "manhattan":
            return calcul_distance.manhattan()
        elif distance == "euclidienne":
            return calcul_distance.euclidean()
        elif distance == "chebyshev":
            return calcul_distance.chebyshev()
        elif distance == "intersection":
            return calcul_distance.intersection()
        elif distance == "khi-2":
            return calcul_distance.khi2()
        elif distance == "minowski":
            return calcul_distance.minowski()
        return np.inf
    
    
    def use_color_descriptor(self, image, color_descriptor, espace_color, canaux_rgb_indexe=(2,2,2), interval=3, dim_fen=4):
        """
            Utiliser un descripteur de couleur
            Usage:
                use_color_descriptor(image, color_descriptor, espace_color, canaux_rgb_indexe, interval, dim_fen)
            Returns:
                np.array - Descripteur de couleur
        """
        if color_descriptor == "Blobs":
            return DescripteurCouleurs.blob(image, espace_color, interval=interval, dim_fen=dim_fen, canaux_rgb_indexe=canaux_rgb_indexe)
        
        
    def use_cnn_descriptor(self, image, cnn_descriptor, cnn_model=None):
        """
            Utiliser un descripteur CNN
            Usage:
                use_cnn_descriptor(image, cnn_descriptor, cnn_model)
            Returns:
                np.array - Descripteur CNN
        """
        if cnn_descriptor is not None:
            return cnn_model.extract_features(image, type=cnn_descriptor)
        return np.array([])

    def mean_average_precision(self, image, images):
        """
            Calcul de la précision moyenne
            Usage:
                mean_average_precision(image, images)
            Returns:
                float - Moyenne de la précision moyenne
        """
        precision = 0
        index = 1
        for i, img in enumerate(images):
            if image[:7] == img['name'][:7]:
                if image != img['name']:
                    precision += index / i
                    index += 1
        return precision / 4

    def get_mean_average_precision(self, limit=10):
        """
            Récupérer la précision moyenne
            Usage:
                get_mean_average_precision(limit)
            Returns:
                list - Précision moyenne
        """
        data = self.db_connect.select_with_orderby_and_limit(
            'precisions',
            columns=['distance', 'color_descriptor', 'espace_color', 'nomalisation', 'shape_descriptor', 'filter', 'texture_descriptor', 'cnn_descriptor', 'p_minowski', 'canal_r', 'canal_g', 'canal_b', 'dim_fen', 'interval', 'precision'],
            orderby='precision',
            limit=limit
        )
        self.precisions =  convert_to_dataframe(data, ['distance', 'color_descriptor', 'espace_color', 'nomalisation', 'shape_descriptor', 'filter', 'texture_descriptor', 'cnn_descriptor', 'p_minowski', 'canal_r', 'canal_g', 'canal_b', 'dim_fen', 'interval', 'precision'])
                    
            
            
    