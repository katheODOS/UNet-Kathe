Hello! Here you'll see that there are different files involved in UNet-Kathe training. 

_train.py_: runs the standard Dice loss function on the data of your choosing. 

_train_cross_entropy.py_ runs the UNet-Kathe with a weighted cross-entropy loss function instead of the Dice loss function.

_hyperparameter_tuning.py_ runs hyperparameter tuning and trains the UNet-Kathe with various datasets and hyperparameter configurations on the Dice loss function. 

_hyperparameter_tuning_cross_entropy.py_ runs hyperparameter tuning and trains the UNet-Kathe with various datasets and hyperparameter configurations on the weighted cross-entropy loss function. 

_run_specific_configs.py_ runs batch trainngs for optimal runs to determine their F1 scores and overall accuracies. Used after running the hyperparameter search and having narrowed down the ideal model configurations.
