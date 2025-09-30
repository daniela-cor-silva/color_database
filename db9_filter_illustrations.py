import os
import shutil
import argparse

# args
parser = argparse.ArgumentParser(description="filter full bodies")
parser.add_argument("--labels_folder", type=str, required=True, help="yolo labels")
parser.add_argument("--all_bird_colors", type=str, required=True, help="color data files")
parser.add_argument("--full_body_birds", type=str, required=True, help="output filtered images")
args = parser.parse_args()

labels_folder = args.labels_folder
all_bird_colors = args.all_bird_colors
full_body_birds = args.full_body_birds

full_birds = []

for file in os.listdir(all_bird_colors):
    name = ('_').join(file.split('_')[3:])
    if name == 'Chaetura_vauxi_richmondi_Group_153391151.txt':
        name = 'Chaetura_vauxi_[richmondi_Group]_153391151.txt'
    label_path = os.path.join(labels_folder, name)
    with open(label_path, 'r') as l:
        eye_beak = False
        paws = False
        for line in l:
            if line[0] == '0' or line[0] == '1':
                eye_beak = True
            elif line[0] == '2':
                paws = True
        if eye_beak == True and paws == True:
            full_birds.append(file)

for bird in full_birds:
    source = os.path.join(all_bird_colors, bird)
    print('Source path:')
    print(source)
    print('-------------------------')
    destination = os.path.join(full_body_birds, bird)
    print('Destination path:')
    print(destination)
    print('-------------------------')
    shutil.copy(source, destination)

