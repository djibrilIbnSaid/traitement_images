# Projet de Traitement d'Images

## Description
Ce projet est dédié au traitement d'images en utilisant diverses techniques et algorithmes pour améliorer, analyser et manipuler des images numériques.

## Installation
1. Cloner le dépôt :
```bash
git clone https://github.com/djibrilIbnSaid/traitement_images.git
```
2. Accéder au répertoire du projet :
```bash
cd traitement_images
```
3. Créer et activer un environnement virtuel :
```bash
python3 -m venv .
source bin/activate
```
4. Créer ou modifier le fichier `.env` à la racine du projet et ajouter les variables d'environnement :
```bash
DB_IMAGE_PATH=chemin_de_la_base_de_données
```
5. Installer les dépendances :
```bash
pip install -r requirements.txt
```
6. Exécuter le script principal :
```bash
python demo.py
```

## Fonctionnalités
- **Déscripteur de couleur** : Extraire les couleurs dominantes d'une image.
- **Déscripteur de fome** : Extraire les formes d'une image.
- **Déscripteur de texture** : Extraire les textures d'une image.
- **Déscripteur CNN** : Extraire les caractéristiques d'une image en utilisant un réseau de neurones convolutif.
- **Calculer la distance des vecteurs d'image** : Calculer la distance entre deux vecteurs d'images.

## Prérequis
- Python 3.x
- Bibliothèques Python : `numpy`, `gradio`, `numpy`, `pytorch`, `keras`

## Structure du Projet
- `demo.py` : Script principal pour exécuter les traitements d'images.
- `mean_average_precision.py` : Calculer la précision moyenne pour évaluer les performances des modèles.
- `db/` : Contient le code pour la base de données.
- `core/` : Contient les classes et fonctions pour les traitements d'images.

## Licence
Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## Auteurs
- **Abdoulaye Djibril DIALLO** - *Etudiant* - [Profil GitHub](https://github.com/djibrilIbnSaid)
