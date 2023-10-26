import os
from PIL import Image
import pandas as pd

def get_images_path_and_name(path = "./images/"):
    df = pd.DataFrame(columns=["path", "name"])

    images = []

    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        name, filetype = filename.split('.')

        if filetype != "jpg":
            im = Image.open(os.path.join(path, filename))
            os.remove(os.path.join(path, filename))
            rgb_im = im.convert("RGB")
            filename = f"{name.split('.')[0]}.jpg"
            rgb_im.save(filepath)

        nombre, apellido, idx = name.split("_")

        image = {"path": os.path.join(path, filename),
                "name": f"{nombre.capitalize()} {apellido.capitalize()}"}
        
        images.append(image)
        
    df = pd.DataFrame.from_records(images)
    
    return df