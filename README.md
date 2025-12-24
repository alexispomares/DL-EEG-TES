## Deep Learning classification of EEG responses to TES
This repository contains a framework for leveraging Deep Learning (DL) algorithms to classify Electroencephalography (EEG) brain responses evoked by Transcranial Electrical Stimulation (TES).

As of 2026, this research work is now published on arXiv: [arxiv.org/pdf/2512.20319](https://arxiv.org/pdf/2512.20319)

More extensive documentation can be found in my [MRes Neurotechnology thesis](https://drive.google.com/file/d/1LB9YoMJ4HiKsqeFQbf2XXOco3DdwnjdF/view) at Imperial College, London.

Our neuroscientific EEG-TES datasets used to train the DL models can be found in Kaggle, as [Raw MFF](https://www.kaggle.com/alexispomares/dissertation-raw) and [Preprocessed CSV](https://www.kaggle.com/alexispomares/dissertation-preprocessed) files.


## Usage
Clone the repo to your local/cloud machine. Download the EEG-TES data from Kaggle into the root *data* folder. If memory is a concern, feel free to choose either dataset depending on the desired pipeline: [raw](https://www.kaggle.com/alexispomares/dissertation-raw) for EEG manipulation; [preprocessed](https://www.kaggle.com/alexispomares/dissertation-preprocessed) for DL classification.

Both pipelines are located inside the root *code* folder, with accompanying Jupyter notebooks that illustrate in detail how to use the modules.


## Where do the EEG-TES datasets come from?
We enrolled 11 healthy resting awake participants (4 female; ages 20-37, average 25.0±4.6 years old) to conduct 13 separate experimental sessions (subjects *P000* and *P001* participated twice, after results from first sessions were found invalid). Participants were instructed to sit awake with eyes open, and blinded to conditions applied. Following an initial rest period of 120 seconds (including 60 seconds with eyes closed), up to 58 blocks of TES were performed, with a total time of up to ~60 minutes per session as tolerated per the participant. EEG was continuously recorded at 1000Hz, and later resampled to 250Hz in our [MNE](https://mne.tools/stable/index.html) preprocessing pipeline.

For equipment we used a newly-acquired GTEN 200 neuromodulation system (Electrical Geodesics, Inc.) that allows simultaneously delivering TES and recording high-density EEG through the same 256-electrodes cap (see Figure below). We delivered tDCS and tACS stimulation across 2 cortical regions: bilateral posterior (targeting angular gyrus) and bilateral frontal (middle frontal gyrus).

![Figure 1](https://github.com/alexispomares/DL-EEG-TES/blob/main/support-data/pics/Figure1.png?raw=true)


## How does the DL pipeline work?
It digests the input data folder (*timeseries* or *features* CSV files), splits into Train/Validation & Holdout *[tf.data.Datasets](https://www.tensorflow.org/api_docs/python/tf/keras/utils/timeseries_dataset_from_array)*, trains our CNN models in [TensorFlow](https://www.tensorflow.org/), and evaluates model performance & interpretability.

![Figure 2](https://github.com/alexispomares/DL-EEG-TES/blob/main/support-data/pics/Figure2.png?raw=true)

Show below are the results for our best CNN algorithm, which outperformed other architectures such as GRU & LSTM RNNs and VGG-16. Top f1-score accuracies were 93% for the Training dataset & 68% for Holdout data.

![Figure 3](https://github.com/alexispomares/DL-EEG-TES/blob/main/support-data/pics/Figure3.png?raw=true)


## Acknowledgments
Thank you to all my participants for playing a crucial part in this study, enabling the creation of two public datasets to be used freely by the DL-EEG research community.

Special thanks for their expert guidance during this research to:  
*Dr. Ines Ribeiro Violante*  
*Dr. Gregory Scott*  


## Questions?
Raise a [GitHub Issue](https://github.com/alexispomares/DL-EEG-TES/issues) or contact me via [LinkedIn](https://www.linkedin.com/in/alexispomares/).

Thanks,  
[Alexis Pomares](https://www.linkedin.com/in/alexispomares/)
