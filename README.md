# Introduction 
This repository contains the code that has been used in the thesis research made by Olivier Spliethoff. 
This thesis focuses on developing a computer vision algorithm to detect and segment spark erosion regions of Insulated Rail Joints (IRJs). 
It is conducted in collaboration with ProRail, a private limited liability company of which the Dutch government is the sole shareholder. 
This company is responsible for maintaining, building and operating the railway network in the Netherlands. 
The main goal of this research is to establish to what extent it is possible to use computer vision algorithms for the detection and quantification of the spark erosion regions in IRJs. 
This study investigates if convolutional neural networks (CNN) are capable of quantifying and detecting spark erosion regions in IRJs. 
More specifically, versions of the U-net, ResU-net, RU-net and R2U-net algorithms were tested and evaluated for using this repository.

# Getting Started
In order to use this repository the dependencies have to be installed.
These dependencies are added in the requirements.txt.
The following libraries have to be installed: 

python==3.7.7
tqdm==4.43.0
numpy==1.18.1
opencv-python==4.2.0.32
scikit-learn==0.23.1
tensorflow-cpu==2.2.0
scikit-metrics==0.1.0


# Build and Test
## Training a model 
The folder data contains the dataset used for detecting spark erosion regions in IRJs. 
You can train an algorithm by running the file train_algorithm_crossvalidation.py
You can choose choose what algorithm you want to train by editing the model = unet. line in this file. 
There as explained in the thesis there are 5 choices between algorithms. 

	1. the baseline algorithm: model = unet.baseline_CNN()  
	2. the U-net algorithm: model = unet.U_net_model()
	3. the ResU-net algorithm: model = unet.Resu_net_model()
	4. the RU-net algorithm: model = unet.RU_net_model()
	5. the R2U-net algorithm: model = unet.R2U_net_model()

You can also change the amount of Epoch and batch size further down in the code. 
Training will be done using the 5 fold cross validation technique. 
This will result into 5 different training sessions using 5 different splits of the training data. 
When running the program creates 5 different sub folders in Sparkerosie/data/Training_session.
This folder will contain the validation part of the data together with a CSV of the training and validation loss during training.

The the models will be saved in the sub folder: Sparkerosie/models.
Each time the validation loss improves the model will be saved.
The filename under which the model will be saved beginning with the session number and ending with the epoch and validation loss. 

## Validating a model 
In order to validate the models on the testset the file Evaluate_crossvalidation_algorithm.py has to be run. 
In this file you have to specify which models need to be evaluated. 
The evaluation models should be in the Sparkerosie/models folder. 
When running Evaluate_crossvalidation_algorithm.py will create a CSV in the in each of subfolders Sparkerosie/data/Training_session created when training the algorithm. 
This CSV will contain all the evaluation metrics per testset picture.

## Displaying and checking the results of the algorithm
In order to display the results of the trained algorithm the file Show_prediction_algorithm.py should be run. 
This function will upload the results of the prediction on the testset in the Sparkerosie/data/Results sub folder. 
The function will also display each of the 40 different testresults on the computer screen for evaluation. 

# Contribute
As of yet it is not possible to do any contributions. 
This will change in the future. 


