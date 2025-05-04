Hello! This README file covers all relevant information about this subdirectory which contains files that will help you preprocess your data after creating and tiling it in QGIS.

For a guide on how to do that please see the flowchart below:

![image](https://github.com/user-attachments/assets/ba9d6679-1970-4887-b33b-6ac745fae716)


After doing that, you can use some of the preprocessing scripts in this directory (and subdirectories), although depending on your needs not all scripts will be necessary. I just included the ones I needed to fully analyze and process my data for the biodiversity dataset. 

Attached below you will find a simple flow chart that shows how some of these files can be used in succession, and some are scattered between directories as the directories group the files based on use case instead of a simple flow. This is to say some scripts can be found in different folders but the names are all the same. 

![image](https://github.com/user-attachments/assets/9518da3a-f264-46ab-9520-0ad40325f5e1)

Here are brief explanations you can find of each subfolder and the kinds of files you will see in them:

_preprocessing_scripts/copying_files_: copies files into folders depending on your directory structure to work with copies instead of originals. This helps you exercise version control and fall back on copies you have not edited in the case of an error. Also, this helps when standardizing the files across datasets (so making sure Dataset B and Dataset B SA have the same files for example). 

_preprocessing_scripts/data_augmentation_: has files that can help you augment your files based on the specific needs of your dataset. Below you can find a flowchart that describes the process for creating Dataset D (but you can modify the steps if necessary. For example, selecting the files to augment, or if you have to implement a different background color for transparent pixels, etc.) Find a general flowchart of how to use those files below:

![image](https://github.com/user-attachments/assets/2a4b098d-6e71-4342-9d62-9cf3a7d5aaee)

_preprocessing_scripts/explore_tifs_: this is what I used to do statistical analysis of the tifs and figure out which ones had valuable information for me, analyze the relative frequencies of pixels across all tifs, and gain a better mathematical idea of how the labels were distributed and which labels were seen in conjunction with other labels.

_preprocessing_scripts/tif_to_png_: used for conversion from the tif format into a .png which is compatible with the UNet-Kathe.
