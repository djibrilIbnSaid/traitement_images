import os

from core.utils import get_images
from core.traitement import Traitement

class MeanAvaragePrecision:
    def __init__(self):
        self.traitement = Traitement()
    
    def apply(self):
        """
            Applique la méthode Mean Average Precision
            Usage:
                MeanAvaragePrecision.apply()
            Arguments:
                None
            Returns:
                None
        """
        self.traitement.db_connect.offline = True
        print("Lancement de la méthode Mean Average Precision")
        distances = ["euclidienne", "manhattan", "chebyshev", "khi-2", "minowski"]
        colors = {
            "2D": ["gray basic", "gray 709", "gray 601", "rgb indexe", "hsv indexe", "hsl indexe"],
            "3D": ["rgb", "rgb normalized", "yiq", "yuv", "l1l2l3", "hsv", "hsl", "lab", "luv"],
            "4D": ["cmyk"],
            "H pondere par S": ["hsv", "hsl"],
            "Blobs": ["gray basic", "gray 709", "gray 601", "rgb indexe", "hsv indexe", "hsl indexe"]
        }
        color_descriptors = ["2D", "3D", "4D", "H pondere par S", "Blobs"]
        # color_descriptors = ["4D", "H pondere par S", "Blobs"]
        normalisations = ["Occurence", "Frequence", "Statistique", "MinMax", "Rang"]
        # normalisations = ["Occurence", "Frequence", "Statistique"]
        descripteur_formes = ["HOG", "HOPN", "HBO", "HBOQ"]
        filters = ["Sobel", "Prewitt", "Scharr"]
        descripteur_textures = ["LBP", "Statistique", "GLCM"]
        # cnn_descriptors = ["VGG16", "MobileNet"]
        cnn_descriptors = ["VGG16"]
        
        images = get_images(self.traitement.path_images)
        
        
        # A supprimer
        distances = ["euclidienne", "manhattan"]
        colors = {
            "2D": ["rgb indexe"]
        }
        color_descriptors = ["2D"]
        normalisations = ["Occurence", "Frequence", "Statistique", "MinMax", "Rang"]
        descripteur_textures = ["LBP"]
        for discriptor in color_descriptors:
            for espace_color in colors[discriptor]:
                for normalisation in normalisations:
                    for desc_texture in descripteur_textures:
                        for distance in distances:
                            precision = 0
                            print(f"Discriptor: {discriptor}, Color: {espace_color}, Normalisation: {normalisation}, Distance: {distance}")
                            try:
                                for filename in images:
                                    p, id = self.traitement.recherche_images(base_image=filename, distance=distance, color_descriptor=discriptor, espace_color=espace_color, nomalisation=normalisation, texture_descriptor=desc_texture, maen_average_precision=True)
                                    precision += p
                                precision /= len(images)
                                self.save_map(id, distance=distance, color_descriptor=discriptor, espace_color=espace_color, normalisation=normalisation, precision=precision)
                            except Exception as e:
                                self.log_exception(f"Discriptor: {discriptor}, Color: {espace_color}, Normalisation: {normalisation}, Distance: {distance}", e)
            
            
        
        
        
        
        # # Parcourir les discripteurs de couleur
        # for discriptor in color_descriptors:
        #     for espace_color in colors[discriptor]:
        #         for normalisation in normalisations:
        #             for distance in distances:
        #                 precision = 0
        #                 print(f"Discriptor: {discriptor}, Color: {espace_color}, Normalisation: {normalisation}, Distance: {distance}")
        #                 try:
        #                     for filename in images:
        #                         p, id = self.traitement.recherche_images(base_image=filename, distance=distance, color_descriptor=discriptor, espace_color=espace_color, nomalisation=normalisation, maen_average_precision=True)
        #                         precision += p
        #                     precision /= len(images)
        #                     self.save_map(id, distance=distance, color_descriptor=discriptor, espace_color=espace_color, normalisation=normalisation, precision=precision)
        #                 except Exception as e:
        #                     self.log_exception(f"Discriptor: {discriptor}, Color: {espace_color}, Normalisation: {normalisation}, Distance: {distance}", e)
        
        # # Parcourir les discripteurs de forme
        # for discriptor in descripteur_formes:
        #     for filter in filters:
        #         for distance in distances:
        #             precision = 0
        #             print(f"Discriptor: {discriptor}, Filter: {filter}, Distance: {distance}")
        #             try:   
        #                 for filename in images:
        #                     p, id = self.traitement.recherche_images(base_image=filename, distance=distance, shape_descriptor=discriptor, filter=filter, maen_average_precision=True)
        #                     precision += p
        #                 precision /= len(images)
        #                 self.save_map(id, distance=distance, shape_descriptor=discriptor, filter=filter, precision=precision)
        #             except Exception as e:
        #                 self.log_exception(f"Discriptor: {discriptor}, Filter: {filter}, Distance: {distance}", e)
        
        # # Parcourir les discripteurs de texture
        # for discriptor in descripteur_textures:
        #     for distance in distances:
        #         precision = 0
        #         print(f"Discriptor: {discriptor}, Distance: {distance}")
        #         try:
        #             for filename in images:
        #                 p, id =  self.traitement.recherche_images(base_image=filename, distance=distance, texture_descriptor=discriptor, maen_average_precision=True)
        #                 precision += p
        #             precision /= len(images)
        #             self.save_map(id, distance=distance, texture_descriptor=discriptor, precision=precision)
        #         except Exception as e:
        #             self.log_exception(f"Discriptor: {discriptor}, Distance: {distance}", e)
        
        # # Parcourir les discripteurs CNN
        # for discriptor in cnn_descriptors:
        #     for distance in distances:
        #         precision = 0
        #         print(f"Discriptor: {discriptor}, Distance: {distance}")
        #         try:
        #             for filename in images:
        #                 p, id = self.traitement.recherche_images(base_image=filename, distance=distance, cnn_descriptor=discriptor, maen_average_precision=True)
        #                 precision += p
        #             precision /= len(images)
        #             self.save_map(id, distance=distance, cnn_descriptor=discriptor, precision=precision)
        #         except Exception as e:
        #             self.log_exception(f"Discriptor: {discriptor}, Distance: {distance}", e)
        
        # # Parcourir les descripteurs de couleur, et de forme
        # for color_discriptor in color_descriptors:
        #     for espace_color in colors[color_discriptor]:
        #         for normalisation in normalisations:
        #             for shape_discriptor in descripteur_formes:
        #                 for filter in filters:
        #                     for distance in distances:
        #                         precision = 0
        #                         print(f"Color: {color_discriptor}, Espace color: {espace_color}, Normalisation: {normalisation}, Shape: {shape_discriptor}, Filter: {filter}, Distance: {distance}")
        #                         try:
        #                             for filename in images:
        #                                 p, id = self.traitement.recherche_images(base_image=filename, distance=distance, color_descriptor=color_discriptor, espace_color=espace_color, nomalisation=normalisation, shape_descriptor=shape_discriptor, filter=filter, maen_average_precision=True)
        #                                 precision += p
        #                             precision /= len(images)
        #                             self.save_map(id, distance=distance, color_descriptor=color_discriptor, espace_color=espace_color, nomalisation=normalisation, shape_descriptor=shape_discriptor, filter=filter, precision=precision)
        #                         except Exception as e:
        #                             self.log_exception(f"Color: {color_discriptor}, Espace color: {espace_color}, Normalisation: {normalisation}, Shape: {shape_discriptor}, Filter: {filter}, Distance: {distance}", e)
        
        # # Parcourir les descripteurs de couleur, et de texture
        # for color_discriptor in color_descriptors:
        #     for espace_color in colors[color_discriptor]:
        #         for normalisation in normalisations:
        #             for texture_discriptor in descripteur_textures:
        #                 for distance in distances:
        #                     precision = 0
        #                     print(f"Color: {color_discriptor}, Espace color: {espace_color}, Normalisation: {normalisation}, Texture: {texture_discriptor}, Distance: {distance}")
        #                     try:
        #                         for filename in images:
        #                             p, id = self.traitement.recherche_images(base_image=filename, distance=distance, color_descriptor=color_discriptor, espace_color=espace_color, nomalisation=normalisation, texture_descriptor=texture_discriptor, maen_average_precision=True)
        #                             precision += p
        #                         precision /= len(images)
        #                         self.save_map(id, distance=distance, color_descriptor=color_discriptor, espace_color=espace_color, nomalisation=normalisation, texture_descriptor=texture_discriptor, precision=precision)
        #                     except Exception as e:
        #                         self.log_exception(f"Color: {color_discriptor}, Espace color: {espace_color}, Normalisation: {normalisation}, Texture: {texture_discriptor}, Distance: {distance}", e)
        
        
        # # Parcourir les descripteurs de couleur, et de CNN
        # for color_discriptor in color_descriptors:
        #     for espace_color in colors[color_discriptor]:
        #         for normalisation in normalisations:
        #             for cnn_discriptor in cnn_descriptors:
        #                 for distance in distances:
        #                     precision = 0
        #                     print(f"Color: {color_discriptor}, Espace color: {espace_color}, Normalisation: {normalisation}, CNN: {cnn_discriptor}, Distance: {distance}")
        #                     try:
        #                         for filename in images:
        #                             p, id = self.traitement.recherche_images(base_image=filename, distance=distance, color_descriptor=color_discriptor, espace_color=espace_color, nomalisation=normalisation, cnn_descriptor=cnn_discriptor, maen_average_precision=True)
        #                             precision += p
        #                         precision /= len(images)
        #                         self.save_map(id, distance=distance, color_descriptor=color_discriptor, espace_color=espace_color, nomalisation=normalisation, cnn_descriptor=cnn_discriptor, precision=precision)
        #                     except Exception as e:
        #                         self.log_exception(f"Color: {color_discriptor}, Espace color: {espace_color}, Normalisation: {normalisation}, CNN: {cnn_discriptor}, Distance: {distance}", e)
        
        # # Parcourir les descripteurs de forme, et de texture
        # for shape_discriptor in descripteur_formes:
        #     for filter in filters:
        #         for texture_discriptor in descripteur_textures:
        #             for distance in distances:
        #                 precision = 0
        #                 print(f"Shape: {shape_discriptor}, Filter: {filter}, Texture: {texture_discriptor}, Distance: {distance}")
        #                 try:   
        #                     for filename in images:
        #                         p, id = self.traitement.recherche_images(base_image=filename, distance=distance, shape_descriptor=shape_discriptor, filter=filter, texture_descriptor=texture_discriptor, maen_average_precision=True)
        #                         precision += p
        #                     precision /= len(images)
        #                     self.save_map(id, distance=distance, shape_descriptor=shape_discriptor, filter=filter, texture_descriptor=texture_discriptor, precision=precision)
        #                 except Exception as e:
        #                     self.log_exception(f"Shape: {shape_discriptor}, Filter: {filter}, Texture: {texture_discriptor}, Distance: {distance}", e)
                            
        # # Parcourir les descripteurs de forme, et de CNN
        # for shape_discriptor in descripteur_formes:
        #     for filter in filters:
        #         for cnn_discriptor in cnn_descriptors:
        #             for distance in distances:
        #                 precision = 0
        #                 print(f"Shape: {shape_discriptor}, Filter: {filter}, CNN: {cnn_discriptor}, Distance: {distance}")
        #                 try:
        #                     for filename in images:
        #                         p, id = self.traitement.recherche_images(base_image=filename, distance=distance, shape_descriptor=shape_discriptor, filter=filter, cnn_descriptor=cnn_discriptor, maen_average_precision=True)
        #                         precision += p
        #                     precision /= len(images)
        #                     self.save_map(id, distance=distance, shape_descriptor=shape_discriptor, filter=filter, cnn_descriptor=cnn_discriptor, precision=precision)
        #                 except Exception as e:
        #                     self.log_exception(f"Shape: {shape_discriptor}, Filter: {filter}, CNN: {cnn_discriptor}, Distance: {distance}", e)
                            
        # # Parcourir les descripteurs de texture, et de CNN
        # for texture_discriptor in descripteur_textures:
        #     for cnn_discriptor in cnn_descriptors:
        #         for distance in distances:
        #             precision = 0
        #             print(f"Texture: {texture_discriptor}, CNN: {cnn_discriptor}, Distance: {distance}")
        #             try:
        #                 for filename in images:
        #                     p, id = self.traitement.recherche_images(base_image=filename, distance=distance, texture_descriptor=texture_discriptor, cnn_descriptor=cnn_discriptor, maen_average_precision=True)
        #                     precision += p
        #                 precision /= len(images)
        #                 self.save_map(id, distance=distance, texture_descriptor=texture_discriptor, cnn_descriptor=cnn_discriptor, precision=precision)
        #             except Exception as e:
        #                 self.log_exception(f"Texture: {texture_discriptor}, CNN: {cnn_discriptor}, Distance: {distance}", e)
        
        # # Parcourir les descripteurs de couleur, de forme, et de texture
        # for color_discriptor in color_descriptors:
        #     for espace_color in colors[color_discriptor]:
        #         for normalisation in normalisations:
        #             for shape_discriptor in descripteur_formes:
        #                 for filter in filters:
        #                     for texture_discriptor in descripteur_textures:
        #                         for distance in distances:
        #                             precision = 0
        #                             print(f"Color: {color_discriptor}, Espace color: {espace_color}, Normalisation: {normalisation}, Shape: {shape_discriptor}, Filter: {filter}, Texture: {texture_discriptor}, Distance: {distance}")
        #                             try:
        #                                 for filename in images:
        #                                     p, id = self.traitement.recherche_images(base_image=filename, distance=distance, color_descriptor=color_discriptor, espace_color=espace_color, nomalisation=normalisation, shape_descriptor=shape_discriptor, filter=filter, texture_descriptor=texture_discriptor, maen_average_precision=True)
        #                                     precision += p
        #                                 precision /= len(images)
        #                                 self.save_map(id, distance=distance, color_descriptor=color_discriptor, espace_color=espace_color, nomalisation=normalisation, shape_descriptor=shape_discriptor, filter=filter, texture_descriptor=texture_discriptor, precision=precision)
        #                             except Exception as e:
        #                                 self.log_exception(f"Color: {color_discriptor}, Espace color: {espace_color}, Normalisation: {normalisation}, Shape: {shape_discriptor}, Filter: {filter}, Texture: {texture_discriptor}, Distance: {distance}", e)
                                        
        # # Parcourir les descripteurs de couleur, de forme, et de CNN
        # for color_discriptor in color_descriptors:
        #     for espace_color in colors[color_discriptor]:
        #         for normalisation in normalisations:
        #             for shape_discriptor in descripteur_formes:
        #                 for filter in filters:
        #                     for cnn_discriptor in cnn_descriptors:
        #                         for distance in distances:
        #                             precision = 0
        #                             print(f"Color: {color_discriptor}, Espace color: {espace_color}, Normalisation: {normalisation}, Shape: {shape_discriptor}, Filter: {filter}, CNN: {cnn_discriptor}, Distance: {distance}")
        #                             try:
        #                                 for filename in images:
        #                                     p, id = self.traitement.recherche_images(base_image=filename, distance=distance, color_descriptor=color_discriptor, espace_color=espace_color, nomalisation=normalisation, shape_descriptor=shape_discriptor, filter=filter, cnn_descriptor=cnn_discriptor, maen_average_precision=True)
        #                                     precision += p
        #                                 precision /= len(images)
        #                                 self.save_map(id, distance=distance, color_descriptor=color_discriptor, espace_color=espace_color, nomalisation=normalisation, shape_descriptor=shape_discriptor, filter=filter, cnn_descriptor=cnn_discriptor, precision=precision)
        #                             except Exception as e:
        #                                 self.log_exception(f"Color: {color_discriptor}, Espace color: {espace_color}, Normalisation: {normalisation}, Shape: {shape_discriptor}, Filter: {filter}, CNN: {cnn_discriptor}, Distance: {distance}", e)
        
        # # Parcourir les descripteurs de couleur, de texture, et de CNN
        # for color_discriptor in color_descriptors:
        #     for espace_color in colors[color_discriptor]:
        #         for normalisation in normalisations:
        #             for texture_discriptor in descripteur_textures:
        #                 for cnn_discriptor in cnn_descriptors:
        #                     for distance in distances:
        #                         precision = 0
        #                         print(f"Color: {color_discriptor}, Espace color: {espace_color}, Normalisation: {normalisation}, Texture: {texture_discriptor}, CNN: {cnn_discriptor}, Distance: {distance}")
        #                         try:
        #                             for filename in images:
        #                                 p, id = self.traitement.recherche_images(base_image=filename, distance=distance, color_descriptor=color_discriptor, espace_color=espace_color, nomalisation=normalisation, texture_descriptor=texture_discriptor, cnn_descriptor=cnn_discriptor, maen_average_precision=True)
        #                                 precision += p
        #                             precision /= len(images)
        #                             self.save_map(id, distance=distance, color_descriptor=color_discriptor, espace_color=espace_color, nomalisation=normalisation, texture_descriptor=texture_discriptor, cnn_descriptor=cnn_discriptor, precision=precision)
        #                         except Exception as e:
        #                             self.log_exception(f"Color: {color_discriptor}, Espace color: {espace_color}, Normalisation: {normalisation}, Texture: {texture_discriptor}, CNN: {cnn_discriptor}, Distance: {distance}", e)
                                    
        # # Parcourir les descripteurs de forme, de texture, et de CNN
        # for shape_discriptor in descripteur_formes:
        #     for filter in filters:
        #         for texture_discriptor in descripteur_textures:
        #             for cnn_discriptor in cnn_descriptors:
        #                 for distance in distances:
        #                     precision = 0
        #                     print(f"Shape: {shape_discriptor}, Filter: {filter}, Texture: {texture_discriptor}, CNN: {cnn_discriptor}, Distance: {distance}")
        #                     try:
        #                         for filename in images:
        #                             p, id = self.traitement.recherche_images(base_image=filename, distance=distance, shape_descriptor=shape_discriptor, filter=filter, texture_descriptor=texture_discriptor, cnn_descriptor=cnn_discriptor, maen_average_precision=True)
        #                             precision += p
        #                         precision /= len(images)
        #                         self.save_map(id, distance=distance, shape_descriptor=shape_discriptor, filter=filter, texture_descriptor=texture_discriptor, cnn_descriptor=cnn_discriptor, precision=precision)
        #                     except Exception as e:
        #                         self.log_exception(f"Shape: {shape_discriptor}, Filter: {filter}, Texture: {texture_discriptor}, CNN: {cnn_discriptor}, Distance: {distance}", e)
                                
        # # Parcourir tous les discripteurs
        # for color_discriptor in color_descriptors:
        #     for espace_color in colors[color_discriptor]:
        #         for normalisation in normalisations:
        #             for shape_discriptor in descripteur_formes:
        #                 for filter in filters:
        #                     for texture_discriptor in descripteur_textures:
        #                         for cnn_discriptor in cnn_descriptors:
        #                             for distance in distances:
        #                                 precision = 0
        #                                 print(f"Color: {color_discriptor}, Espace color: {espace_color}, Normalisation: {normalisation}, Shape: {shape_discriptor}, Filter: {filter}, Texture: {texture_discriptor}, CNN: {cnn_discriptor}, Distance: {distance}")
        #                                 for filename in images:
        #                                     p, id = self.traitement.recherche_images(base_image=filename, distance=distance, color_descriptor=color_discriptor, espace_color=espace_color, nomalisation=normalisation, shape_descriptor=shape_discriptor, filter=filter, texture_descriptor=texture_discriptor, cnn_descriptor=cnn_discriptor, maen_average_precision=True)
        #                                     precision += p
        #                                 precision /= len(images)
        #                                 self.save_map(distance=distance, color_descriptor=color_discriptor, espace_color=espace_color, nomalisation=normalisation, shape_descriptor=shape_discriptor, filter=filter, texture_descriptor=texture_discriptor, cnn_descriptor=cnn_discriptor, precision=precision)
        
                            
        
    

    
    def save_map(self, id, distance=None, color_descriptor=None, espace_color=None, normalisation=None, shape_descriptor=None, filter=None, texture_descriptor=None, cnn_descriptor=None, p_minowski=None, canal_r=None, canal_g=None, canal_b=None, dim_fen=None, interval=None, precision=None):
        """
            Sauvegarde les résultats de la méthode Mean Average Precision dans la base de données
            Usage:
                MeanAvaragePrecision.save_map()
        """
        self.traitement.db_connect.insert_with_check(
            'precisions',
            [id, distance, color_descriptor, espace_color, normalisation, shape_descriptor, filter, texture_descriptor, cnn_descriptor, p_minowski, canal_r, canal_g, canal_b, dim_fen, interval, precision],
            columns=['id', 'distance', 'color_descriptor', 'espace_color', 'nomalisation', 'shape_descriptor', 'filter', 'texture_descriptor', 'cnn_descriptor', 'p_minowski', 'canal_r', 'canal_g', 'canal_b', 'dim_fen', 'interval', 'precision'],
            col_name='id'
        )

    import os

    def log_exception(self, name: str, exception: Exception, filename: str = "exceptions.log"):
        """
        Enregistre le nom et l'exception dans un fichier.
        - Crée le fichier s'il n'existe pas.
        - Ajoute trois sauts de ligne avant les nouvelles entrées si le fichier contient déjà des données.
        
        :param name: Nom du processus ou de la fonction ayant levé l'exception
        :param exception: L'exception attrapée
        :param filename: Nom du fichier de log (par défaut "exceptions.log")
        """
        # Vérifier si le fichier existe et contient déjà des données
        add_newlines = os.path.exists(filename) and os.path.getsize(filename) > 0
        
        with open(filename, "a", encoding="utf-8") as file:
            if add_newlines:
                file.write("\n\n\n")  # Ajouter trois sauts de ligne
            file.write(f"Nom: {name}\n")
            file.write(f"Exception: {exception}\n")
            file.write("-" * 40 + "\n")
        


if __name__ == "__main__":
    mean_average_precision = MeanAvaragePrecision()
    mean_average_precision.apply()