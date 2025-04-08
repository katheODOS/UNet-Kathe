use qgis to generate the tiles and save them in a folder

in the directory you save them in there will be a folder named ‘1’: delete it the images are lower resolution

rename all the files with the rename_files.py

step 1: create json files with the descending labels from the tifs

step 2: create primary label json

step 3: create secondary label json

step 4: turn the tifs into pngs

step 5: convert color channel from rgba into rgb (convert_to_rgb.py)

step 6: convert image size from 512x512/convert color channel from rgba into rgb (convert_to_rgb.py)

convert_single_channel (when working with single channel dataset)

step 7: random_stratify

step 8: copy_images_into_dataset.py

filter_and_copy_by_labels (create a separate directory where the files to be augmented will be worked with) 

rotate_augmentation.py