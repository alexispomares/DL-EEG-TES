## DL Classification of EEG Responses to TES
This repository contains a framework for leveraging Deep Learning (DL) algorithms to classify EEG brain responses evoked by Transcranial Electrical Stimulation (TES). Extensive documentation can be found in this [MRes Neurotechnology Thesis](https://drive.google.com/file/d/1LB9YoMJ4HiKsqeFQbf2XXOco3DdwnjdF/view).

Our neuroscientific EEG-TES datasets used to train the DL models can be found in Kaggle, as [Raw MFF](https://www.kaggle.com/alexispomares/dissertation-raw) and [Preprocessed CSV](https://www.kaggle.com/alexispomares/dissertation-preprocessed) files.


## Usage
Clone the repo to your local/cloud machine. Download the EEG-TES data from Kaggle into the root *data* folder. If memory is a concern, feel free to choose either dataset depending on the desired pipeline: [raw](https://www.kaggle.com/alexispomares/dissertation-raw) for EEG manipulation; [preprocessed](https://www.kaggle.com/alexispomares/dissertation-preprocessed) for DL classification.

Both pipelines are located inside the root *code* folder, with accompanying Jupyter notebooks that illustrate how to use each file.


## Where do the EEG-TES datasets come from?
We enrolled 11 healthy resting awake participants (4 female; ages 20-37, average 25.0±4.6 years old) to conduct 13 separate experimental sessions (subjects *P000* and *P001* participated twice, after results from first sessions were found invalid). Participants were instructed to sit awake with eyes open, and blinded to conditions applied. Following an initial rest period of 120 seconds (including 60 seconds with eyes closed), up to 58 blocks of TES were performed, with a total time of up to ~60 minutes per session as tolerated per the participant. EEG was continuously recorded at 1000Hz, and later resampled to 250Hz in our preprocessing pipeline.

For equipment we used a newly-acquired GTEN 200 neuromodulation system (Electrical Geodesics, Inc.) that allows simultaneously delivering TES and recording high-density EEG through the same 256-electrodes cap (see Figure below). We delivered tDCS and tACS stimulation across 2 cortical regions: bilateral posterior (targeting angular gyrus) and bilateral frontal (middle frontal gyrus).

![Figure 1](https://github.com/alexispomares/DL-EEG-TES/blob/main/Figure1.png?raw=true)


## How does the DL pipeline work?
It digests the selected input data folder (can be in *timeseries* or in *features* CSV format); splits the data into *[tf.data.Dataset]*(https://www.tensorflow.org/api_docs/python/tf/keras/utils/timeseries_dataset_from_array) instances for Train, Validation & Holdout datasets; trains our Convolutional Neural Network (CNN) models, as illustrated below; evaluates performance & model interpretability.

![Figure 2](https://github.com/alexispomares/DL-EEG-TES/blob/main/Figure2.png?raw=true)

The results for our best CNN algorithm (which significantly outperformed other architectures explored such as GRU & LSTM RNNs, as well as classic image classification models like VGG-16) are shown below. Top f1-score accuracies were 93% for the Training dataset & 68% for Holdout data.

![Figure 3](https://github.com/alexispomares/DL-EEG-TES/blob/main/Figure3.png?raw=true)


## Questions? 
Raise a [GitHub Issue](https://github.com/alexispomares/DL-EEG-TES/issues) or contact me via [LinkedIn](https://www.linkedin.com/in/alexispomares/)!

Thanks,  
[Alexis Pomares Pastor](https://www.linkedin.com/in/alexispomares/)
