Hello! This README file covers all relevant information about this subdirectory which contains files that will help you evaluate the model's performance. 

Attached below you will find a simple flow chart that shows how some of these files can be used in succession. 

![image](https://github.com/user-attachments/assets/e0874b0a-2574-483b-bd14-70305a0ebc28)

Beyond this, you can use the following files for the following purposes:

_predict.py_: generates a prediction for a single validation image based on a model checkpoint of your choosing.

_evaluate_model.py_: to evaluate a single model's class-wise accuracy.

_evaluate_models.py_: to evaluate a set of models' class-wise accuracy.

_generate_predictions.py_: to create masks for all validation images a model was tested on.

_process_overall_accuracies.py_: processes all of the overall accuracies which are extracted from each model's evaluation_report.txt.

_process_validation_scores.py_: processes and sorts all of the models in a checkpoints folder' validation scores through analyzing each model's output.txt for the last Validation Dice score. 
