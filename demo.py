from core.traitement import Traitement
import gradio as gr
import pandas as pd

traitement = Traitement(path_images="BD_images_resized")

discriptors = ["couleur", "forme", "texture", "cnn"]

distances = ["euclidienne", "manhattan", "chebyshev", "intersection", "khi-2", "minowski"]
colors = {
    "2D": ["gray basic", "gray 709", "gray 601", "rgb indexe", "hsv indexe", "hsl indexe"],
    "3D": ["rgb", "rgb normalized", "yiq", "yuv", "l1l2l3", "hsv", "hsl", "lab", "luv"],
    "4D": ["cmyk"],
    "H pondere par S": ["hsv", "hsl"],
    "Blobs": ["gray basic", "gray 709", "gray 601", "rgb indexe", "hsv indexe", "hsl indexe"]
}

color_descriptors = ["2D", "3D", "4D", "H pondere par S", "Blobs"]
normalisations = ["Occurence", "Frequence", "Statistique"]
descripteur_formes = ["HOG", "HOPN", "HBO", "HBOQ"]
filters = ["Sobel", "Prewitt", "Scharr"]
descripteur_textures = ["LBP", "Statistique", "GLCM"]
cnn_descriptors = ["VGG16"]


def toggle_fields(selected_discriptors):    
    return (
        gr.update(visible="couleur" in selected_discriptors),  # color_descriptor
        gr.update(visible="couleur" in selected_discriptors),  # espace_color
        gr.update(visible="forme" in selected_discriptors),    # shape_descriptor
        gr.update(visible="forme" in selected_discriptors),    # filter
        gr.update(visible="texture" in selected_discriptors),  # texture_descriptor
        gr.update(visible="cnn" in selected_discriptors),      # cnn_descriptor
    )

def change_color_descriptor(espace_color):
    """
        Met à jour les choix des espaces de couleurs en fonction du descripteur de couleur sélectionné
        Args:
            espace_color (str): Descripteur de couleur
        Returns:
            gradio.Dropdown: Espace de couleur
    """
    return gr.update(choices=colors.get(espace_color, []))

def update_gallery(nb_lines, nb_columns):
    """
        Met à jour la galerie d'images dynamiquement lors du changement des lignes et colonnes
        Args:
            nb_lines (int): Nombre de lignes
            nb_columns (int): Nombre de colonnes
        Returns:
            gradio.Gallery: Galerie d'images    
    """
    return gr.update(rows=nb_lines, columns=nb_columns)

def afficher_input_p(selection):
    """
        Affiche l'input sélectionné
        Args:
            selection (str): Input sélectionné
        Returns:
            gradio.Number: Input du paramètre p
    """
    return gr.update(visible=selection == "minowski", interactive=True, value=1.5)

def afficher_canneaux_rgb_indexe(color_descriptor):
    """
        Affiche les canneaux RGB indexé en fonction de l'espace couleur
        Args:
            espace_color (str): Espace couleur
        Returns:
            gradio.Number: Canneaux RGB indexé
    """
    check_visible = color_descriptor == "Blobs"
    return (
        gr.update(visible=check_visible, interactive=True, value=2),  # canal_r
        gr.update(visible=check_visible, interactive=True, value=2),  # canal_g
        gr.update(visible=check_visible, interactive=True, value=2),  # canal_b
        gr.update(visible=check_visible, interactive=True, value=3), # dimension de la fenetre
        gr.update(visible=check_visible, interactive=True, value=4), # l'interval des pourcentage       
    )

def afficher_popup(action):
    if action == "Ouvrir":
        return "Bienvenue dans la fenêtre popup !", True
    else:
        return "", False

with gr.Blocks(css="footer {visibility: hidden}", title="INFO-911") as demo:

    gr.Markdown('''
                <h1 style="text-align: center">PROJET INFO 902</h1>
                <h3 style="text-align: center">Recherche d'images par similarité</h3>
                <h5 style="text-align: center"><span>Ce projet a été réalisé par <strong>Abdoulaye Djibril DIALLO</strong></span></h5>
    ''')

    with gr.Row():
        discriptors_selected = gr.CheckboxGroup(discriptors, label="Descripeteus", info="Selectionner les descripteurs")
        offline = gr.Checkbox(label="SQLite", value=True, info="Accès à la base de données", interactive=True)
        mean_average_precision = gr.Checkbox(label="MAP", value=False, info="Mean average precision", interactive=True)
        
    with gr.Row():
        base_image = gr.Image(type="filepath", label="Téléchargez une image ici")

        with gr.Row():
            distance = gr.Dropdown(choices=distances, value=distances[0], label="Distance")
            color_descriptor = gr.Dropdown(choices=color_descriptors, value=color_descriptors[0], label="Discripteur de couleur", visible=False)
            espace_color = gr.Dropdown(choices=colors.get("2D", []), value=colors.get("2D")[0], label="Espace couleur", visible=False)
            nomalisation = gr.Dropdown(choices=normalisations, value=normalisations[0], label="Normalisation")
            shape_descriptor = gr.Dropdown(choices=descripteur_formes, value=descripteur_formes[0], label="Descripteur de forme", visible=False)
            filter = gr.Dropdown(choices=filters, value=filters[0], label="Filtres", visible=False)
            texture_descriptor = gr.Dropdown(choices=descripteur_textures, value=descripteur_textures[0], label="Descripteur de Texture", visible=False)
            cnn_descriptor = gr.Dropdown(choices=["VGG16"], value=cnn_descriptors[0], label="Descripteur CNN", visible=False)
            p = gr.Number(label="Paramètre p de minowski", value=0, minimum=0, visible=False)
            canal_r = gr.Number(label="Canal r indexé", value=2, visible=False)
            canal_g = gr.Number(label="Canal g indexé", value=2, visible=False)
            canal_b = gr.Number(label="Canal b indexé", value=2, visible=False)
            dim_fen = gr.Number(label="Dimension de la fenêtre", value=3, minimum=1, maximum=10, visible=False)
            interval = gr.Number(label="Intervalle", value=4, minimum=1, maximum=10, visible=False)

            nb_responses = gr.Number(label="Nombre de réponses", value=5, minimum=1, maximum=10)
            nb_lines = gr.Number(label="Nombre de lignes", value=1, minimum=1, maximum=10)
            nb_columns = gr.Number(label="Nombre de colonnes", value=5, minimum=3, maximum=10)

    with gr.Row():
        similarities = gr.Gallery(label="Results", columns=5, rows=1, height="auto", object_fit="contain")

    with gr.Row():
        clear_button = gr.Button("Annuler", variant="stop")
        process_button = gr.Button("Rechercher", variant="primary")
    
    
    # Met à jour la galerie dynamiquement lors du changement des lignes et colonnes
    nb_lines.change(fn=update_gallery, inputs=[nb_lines, nb_columns], outputs=similarities)
    nb_columns.change(fn=update_gallery, inputs=[nb_lines, nb_columns], outputs=similarities)
    distance.change(fn=afficher_input_p, inputs=[distance], outputs=p)
    color_descriptor.change(fn=afficher_canneaux_rgb_indexe, inputs=[color_descriptor], outputs=[canal_r, canal_g, canal_b, dim_fen, interval])
    
    
    discriptors_selected.change(
        fn=toggle_fields,
        inputs=[discriptors_selected],
        outputs=[color_descriptor, espace_color, shape_descriptor, filter, texture_descriptor, cnn_descriptor]
    )

    color_descriptor.change(
        fn=change_color_descriptor,
        inputs=[color_descriptor],
        outputs=[espace_color]
    )
    
    def process_search(base_image, distance, color_descriptor, espace_color, nomalisation, shape_descriptor, filter, texture_descriptor, cnn_descriptor, p, canal_r, canal_g, canal_b, dim_fen, interval, nb_responses, offline, discriptors_selected):
        if "couleur" not in discriptors_selected:
            color_descriptor = None
            espace_color = None
            canal_r = 2
            canal_g = 2
            canal_b = 2
            dim_fen = 3
            interval = 4
        
        if "forme" not in discriptors_selected:
            shape_descriptor = None
            filter = None
        
        if "texture" not in discriptors_selected:
            texture_descriptor = None
            dim_fen = None
            interval = None
        
        if "cnn" not in discriptors_selected:
            cnn_descriptor = None
        
        return traitement.recherche_images(base_image, distance, color_descriptor, espace_color, nomalisation, shape_descriptor, filter, texture_descriptor, cnn_descriptor, nb_responses, p, canal_r, canal_g, canal_b, dim_fen, interval, offline)

    process_button.click(
        fn=process_search,
        inputs=[base_image, distance, color_descriptor, espace_color, nomalisation, shape_descriptor, filter, texture_descriptor, cnn_descriptor, p, canal_r, canal_g, canal_b, dim_fen, interval, nb_responses, offline, discriptors_selected],
        outputs=similarities
    )
    

demo.launch(show_api=False)
