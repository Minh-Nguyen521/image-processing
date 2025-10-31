import pandas as pd
import os
from pathlib import Path

translate = {"cane": 1, 
             "cavallo": 2, 
             "elefante": 3, 
             "farfalla": 4, 
             "gallina": 5, 
             "gatto": 6, 
             "mucca": 7, 
             "pecora": 8, 
             "ragno": 9, 
             "scoiattolo": 10, 
             "dog": 1, 
             "cavallo": 2, 
             "elephant" : 3, 
             "butterfly": 4, 
             "chicken": 5, 
             "cat": 6, 
             "cow": 7, 
             "spider": 8, 
             "squirrel": 9
             }

def create_csv(root_dir, output_csv):
    data = []
    root = Path(root_dir)
    for folder in os.listdir(root):
        folder_path = root / folder
        if folder_path.is_dir():
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    data.append({
                        "folder": folder,
                        "image_name": img_name,
                        "animal_label": translate[folder]
                    })
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
if __name__ == "__main__":

    root_directory = "./dataset/raw-img/" 
    output_csv_file = "animals_dataset.csv"
    create_csv(root_directory, output_csv_file)