import tensorflow as tf
from tensorflow.keras import models, layers

from tf_explain.core.grad_cam import GradCAM

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('figure', dpi=200)                                     #set this value if high plot resolutions are desired for HTML/PDF exports

import mne
import os, glob, sys, cmapy, operator, warnings
from IPython.display import display, Javascript

import ipywidgets as widgets
from IPython.display import display, Markdown, Javascript, HTML
from datetime import datetime


##Configure our environment:
hyperparam = {'MINIBATCH_SIZE': 64}


metadata = {'VERSION_NAME': 'Golden_CNN'}

metadata['EEG_RESAMPLING_RATE'] = 250                           # in Hz

metadata['WINDOW_LENGTH']  = 1                                # in seconds
metadata['WINDOW_SIZE']    = metadata['WINDOW_LENGTH'] * metadata['EEG_RESAMPLING_RATE']  + 1    # in samples

metadata['WINDOW_SLIDE']   = 1                                # in seconds
metadata['WINDOW_SHIFT']   = metadata['WINDOW_SLIDE']  * metadata['EEG_RESAMPLING_RATE']  + 1    # in samples

metadata['WINDOW_OVERLAP'] = 0 * metadata['WINDOW_LENGTH']    # in seconds


try:
    if 'google.colab' in str(get_ipython()):                  # if in Colab, further setup environment + necessary files/data
        os.system("pip install -q mne opencv-python cmapy ipympl")
        # os.system("pip install -q -U --force-reinstall --no-deps kaggle")
        
        if not os.getcwd().endswith('cloned-repo'):
            os.system("git clone git://github.com/alexispomares/DL-EEG-TES.git cloned-repo")
            os.system("cd cloned-repo")
            
            os.environ['KAGGLE_USERNAME'] = "yourUsername"
            os.environ['KAGGLE_KEY'] = "yourKey"
            
            os.system("kaggle datasets download -d alexispomares/dissertation-preprocessed --unzip -p data/preprocessed-EEG")     # authenticate & download from Kaggle
except: pass



##Define all our pipeline functions:
def load_EEG_data(data_type, unseen_participant=[], exclude_participants=[], skip_calibration=True, version_name=metadata['VERSION_NAME'], VERBOSE=True):
    assert data_type in ('timeseries', 'features', 'concatenated_features')
    metadata['DATA_TYPE'] = data_type
    metadata['VERSION_NAME'] = version_name
    
    ##Load the CSVs containing our Training EEG data:
    data_pattern = "data/preprocessed-EEG/{}/P???/run*.csv".format(metadata['DATA_TYPE'])
    data_paths = [s.replace('\\', '/') for s in sorted(glob.glob(data_pattern))]
    data = []
    
    excluded = set(exclude_participants + [unseen_participant]) if unseen_participant else set(exclude_participants)
    
    for i, path in enumerate(data_paths.copy()):
        if skip_calibration and '/run0_' in path: data_paths.remove(path)
            
        else: [data_paths.remove(path) for excl in excluded if f'/{excl}/run' in path]
    
    if data_paths:
        metadata['participant_IDs'] = list(sorted(set([p[p.rfind(f"/run") - 4 : p.rfind(f"/run")] for p in data_paths])))

        if VERBOSE: print(f"📢 Reading {len(data_paths)} EEG data files from participants:  {metadata['participant_IDs']}\n")

        for i, filename in enumerate(data_paths):
            data.append(pd.read_csv(filename, header=0))
            if VERBOSE: print(f"Loaded file #{i} => '{filename}'")

        data = pd.concat(data, axis=0, ignore_index=True)
    
    
    ##Load the CSVs containing our Unseen Participant EEG data:
    unseen_pattern = "data/preprocessed-EEG/{}/{}/run*_????????_??????.csv".format(metadata['DATA_TYPE'], unseen_participant)
    unseen_paths = [s.replace('\\', '/') for s in sorted(glob.glob(unseen_pattern))]
    unseen_data = []
    
    if unseen_paths:
        if skip_calibration and '/run0_' in unseen_paths[0]: unseen_paths = unseen_paths[1:]
        
        if VERBOSE: print(f"\n\n📢 Reserving {len(unseen_paths)} EEG data files from '{unseen_participant}' to act as our Unseen Participant:\n")
                
        for i, filename in enumerate(unseen_paths):
            unseen_data.append(pd.read_csv(filename, header=0))
            if VERBOSE: print(f"Loaded file #{i} => '{filename}'")

        unseen_data = pd.concat(unseen_data, axis=0, ignore_index=True)
    
    
    ##Calculate & Print some useful metadata:
    if data_paths:
        metadata['size_samples'] = int(data.shape[0] * (1-metadata['WINDOW_OVERLAP']))
        metadata['size_memory']  = sys.getsizeof(data) / 1024**3

        if metadata['DATA_TYPE'] == 'timeseries':
            metadata['eeg_seconds'] = metadata['size_samples'] / metadata['EEG_RESAMPLING_RATE']
            metadata['channel_labels'] = list(data.columns)[3:]

        elif metadata['DATA_TYPE'] == 'features':
            metadata['n_features'] = int(data[data.groupby('feature').cumcount() == 1].index[0])
            metadata['feature_labels'] = data['feature'][:metadata['n_features']]
            metadata['channel_labels'] = list(data.columns)[3:]

            metadata['eeg_seconds'] = metadata['size_samples'] / metadata['n_features']

        elif metadata['DATA_TYPE'] == 'concatenated_features':
            metadata['feature_labels'] = [f[4:] for f in data.columns if f.startswith('E0:')]
            metadata['n_features'] = len(metadata['feature_labels'])

            metadata['eeg_seconds'] = metadata['size_samples']
    
    
        if VERBOSE:
            print("\n\n\n📢 Loaded our data from a total of {}min ({}h) of EEG, producing {} training examples from {} rows obtained at {}Hz with a {}% overlap."
                  .format(int(metadata['eeg_seconds']/60),  np.round(metadata['eeg_seconds']/60/60, 1),  int(metadata['eeg_seconds']),  metadata['size_samples'],
                  metadata['EEG_RESAMPLING_RATE'],  int(metadata['WINDOW_OVERLAP']*100)))
            
            print("\n📢 The resulting data has a shape of {} and a memory size of {}GB:\n\n".format(data.shape, np.round(metadata['size_memory'], 2)))
            
            if   metadata['DATA_TYPE'] == 'timeseries':            display(data.iloc[metadata['EEG_RESAMPLING_RATE']-3:metadata['EEG_RESAMPLING_RATE']*2+3, :])
            elif metadata['DATA_TYPE'] == 'features':              display(data.iloc[:metadata['n_features']+3, :])
            elif metadata['DATA_TYPE'] == 'concatenated_features': display(data.iloc[:20, :])
    
    
    #Format data for the TensorFlow pipeline:
    metadata['LABEL_MAP'] = {'rest': 0., 'frontal/tACS': 1., 'frontal/tDCS': 2., 'posterior/tACS': 3., 'posterior/tDCS': 4.}
    
    if unseen_paths:
        unseen_data = unseen_data.drop('epoch', axis=1) if metadata['DATA_TYPE'] != 'features' else unseen_data.drop(['epoch', 'feature'], axis=1)
        
        unseen_data['label'] = unseen_data['label'].str.replace(r'.*/2', 'rest', regex=True).replace(r'/1', '', regex=True)
        
        unseen_data['label'].replace(metadata['LABEL_MAP'], inplace=True)

        
    if data_paths:
        if data.isna().sum().sum() > 0: raise Exception("⚠ There are some 'NaN' values in your data!")
        
        data = data.drop('epoch', axis=1) if metadata['DATA_TYPE'] != 'features' else data.drop(['epoch', 'feature'], axis=1)        
        
        data['label'] = data['label'].str.replace(r'.*/2', 'rest', regex=True).replace(r'/1', '', regex=True)    #relabel '.../2' blocks as 'rest' blocks & remove '.../1' tags        
        
        value_counts = pd.concat([data['label'].value_counts(), np.round(data['label'].value_counts()/data['label'].value_counts().sum(), 2)], axis=1).rename_axis('label')
        value_counts.columns = ['number', 'percentage']    
        
        data['label'].replace(metadata['LABEL_MAP'], inplace=True)        
        
        if VERBOSE:
            print('\n\n\n\n\nBrief summary of all data:\n')
            display(data.describe().T)
            
            print('\n')
            display(value_counts)
            
            print('\n\n\nDictionary with Label Mappings:\n')
            display(metadata['LABEL_MAP'])
            
            print('\n\n\n\nFirst few rows of ALL data:\n')
            display(data[:40])
            
            print('\n\n\n\nFirst few rows of UNSEEN data:\n')
            display(unseen_data[:5])
    
    return data, unseen_data, metadata



def create_TF_datasets(data, unseen_data, extra_data=None, split={'train': 0.90, 'val': 0.05, 'test': 0.05}, VERBOSE=True):
    def produce_timeseries_dataset(data):
        if metadata['DATA_TYPE'] == 'timeseries': sequence_length , sequence_stride = metadata['WINDOW_SIZE'], metadata['WINDOW_SHIFT']
        elif metadata['DATA_TYPE'] == 'features': sequence_length = sequence_stride = metadata['n_features']
        elif metadata['DATA_TYPE'] == 'concatenated_features': sequence_length = sequence_stride = 1
        
        tensorflow_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=np.expand_dims(data.drop(['label'], axis=1), -1),
                targets=data['label'].to_numpy(),
                sequence_length=sequence_length,
                sequence_stride=sequence_stride,
                # shuffle=True,    #'reshuffle_each_iteration=False' unfortunately seems to not be an option for this 'shuffle' parameter
                # seed=420,
                batch_size=hyperparam['MINIBATCH_SIZE']
            ).shuffle(data.shape[0], reshuffle_each_iteration=False)
        
        return tensorflow_dataset
    
    
    dataset = {}    
    dataset['ALL']        = produce_timeseries_dataset(data)        
    dataset['Holdout'] = produce_timeseries_dataset(unseen_data)
    
    if extra_data: dataset['Extra'] = produce_timeseries_dataset(extra_data)
    
    metadata['input_shape'] = list(dataset['ALL'])[0][0].numpy().shape
    metadata['n_batches'] = tf.data.experimental.cardinality(dataset['ALL']).numpy()

    if VERBOSE: print("✂️ Created {} batches from the input data, with elements of shape (batch_size, samples, channels, 1) => {}"
                       .format(metadata['n_batches'], metadata['input_shape']))
        

    dataset['Train']      = dataset['ALL'].take(int(split['train'] * metadata['n_batches']))    
    dataset['Validation'] = dataset['ALL'].skip(int(split['train'] * metadata['n_batches'])).take(int(split['val'] * metadata['n_batches']))
    dataset['Test']       = dataset['ALL'].skip(int(split['train'] * metadata['n_batches'])).skip(int(split['val'] * metadata['n_batches']))
    
    del dataset['ALL']
    
    return dataset, metadata



def load_TF_model(load_previous='saved_model', version=-1, VERBOSE=True):    
    if load_previous=='checkpoint_weights':
        path = metadata['checkpoint_path']
        
        model.load_weights(path)
        if VERBOSE: print('Loaded model from: ', path)        
        
    elif load_previous=='checkpoint_model':
        path = metadata['checkpoint_path']
        
        model = tf.keras.models.load_model(path)
        if VERBOSE: print('Loaded model from: ', path)        
        
    elif load_previous=='saved_model':
        path = sorted(glob.glob(metadata['models_path']))[version]
        
        model = tf.keras.models.load_model(path)
        if VERBOSE: print('Loaded model from: ', path)        
        
    elif str(type(load_previous)) == "<class 'tensorflow.python.keras.engine.sequential.Sequential'>": model = load_previous
    
    
    return model



def define_TF_model(load_previous=None, version=-1, VERBOSE=True):    
    ##Define our model architecture according to the Data Type:
    model = models.Sequential(name='{}-CNN'.format(metadata['DATA_TYPE']))
    
    
    if metadata['DATA_TYPE'] == 'timeseries':
            model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='SAME', input_shape=metadata['input_shape'][1:]))
            model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='SAME'))
            model.add(layers.MaxPool2D((2, 2)))
            # model.add(layers.Dropout(.2))

            model.add(layers.Flatten())
            model.add(layers.Dense(64, activation='relu'))
            # model.add(layers.Dropout(.4))
    
    
    
    elif metadata['DATA_TYPE'] == 'features':
        model.add(layers.Conv2D(32, (1, 3), activation='relu', padding='SAME', input_shape=metadata['input_shape'][1:]))
        model.add(layers.Conv2D(64, (1, 3), activation='relu', padding='SAME'))
        # model.add(layers.MaxPool2D((1, 4)))
        # model.add(layers.Dropout(.05))

        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        # model.add(layers.Dropout(.1))
    
    
    elif metadata['DATA_TYPE'] == 'concatenated_features':
        pass # architecture not defined for 'concatenated_features', since it is not relevant to our particular study
    
    
    model.add(layers.Dense(len(metadata['LABEL_MAP']), activation='softmax'))
    
    
    ##Preload and/or Compile our model:
    metadata['checkpoint_path'] = "support-data/DL/model_checkpoints/{}/{}.ckpt".format(metadata['DATA_TYPE'], metadata['VERSION_NAME'])
    metadata['models_path']     = "support-data/DL/model_saved/{}/holdout_acc*-*".format(metadata['DATA_TYPE'])
     
    metadata['checkpoint_callback'] = tf.keras.callbacks.ModelCheckpoint(filepath=metadata['checkpoint_path'],
                                                                         monitor='val_accuracy',
                                                                         save_best_only=True,
                                                                         save_weights_only=False,
                                                                         save_freq='epoch',
                                                                         verbose=1)
    
    if load_previous: model = load_TF_model(load_previous, version, VERBOSE=VERBOSE)    
        
    if load_previous not in ('checkpoint_model', 'saved_model'):
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])
    
    global tf_history
    tf_history = None
    
    return model, metadata



def train_TF_model(model, dataset, epochs=10, VERBOSE=True):
    if VERBOSE:
        model.summary()
        print('\nData Type:        {}\nMinibatch Size:   {}\n{} Participants:  {}\n{} Labels:         {}\n\n{}\n\n\n'
               .format(metadata['DATA_TYPE'], hyperparam['MINIBATCH_SIZE'], len(metadata['participant_IDs']), metadata['participant_IDs'],
                len(metadata['LABEL_MAP']), list(metadata['LABEL_MAP'].keys()), '=' * 110))
    
    temp_history = model.fit(dataset['Train'],
                             validation_data=dataset['Validation'],
                             callbacks=[metadata['checkpoint_callback']],
                             epochs=epochs)
    
    try:
        global tf_history
        
        for k in tf_history.history.keys(): tf_history.history[k] += temp_history.history[k]
        
        tf_history.params['epochs'] += temp_history.params['epochs']
        tf_history.params['steps']  += temp_history.params['steps']
    
    except: tf_history = temp_history    
    
    return tf_history



def compute_importances(model, input_data, to_evaluate='Holdout', choose_label=None, colormaps=[], figwidth=12, VERBOSE=True):
    ##Format our data adequately:
    if type(input_data) == dict: input_data = input_data[to_evaluate]
    
    if str(type(input_data)).startswith("<class 'tensorflow.python.data.ops.dataset_ops"):
        X = np.concatenate([x for x, _ in input_data], axis=0)
        Y = np.concatenate([y for _, y in input_data], axis=0)
    
    elif type(input_data) == tuple:
        X, Y = input_data
        X, Y = [X], [Y]
    
    else: raise Exception("'input_data' is in the wrong format!")
    
    
    ##Explain all examples in our input_data:
    if VERBOSE: print(f'📢 Creating {len(X)} model explainers\n')    
    gradCAM = GradCAM()
    
    explainers = {k: [] for k in metadata['LABEL_MAP'].keys()}
    i = 0
    
    for x, y in zip(X, Y):
        if VERBOSE and i%250 == 0 and len(X) > 250: print(f'Iterated through example #{i}')
        i += 1
        
        label = [k for k,v in metadata['LABEL_MAP'].items() if v == float(y)][0]    #translate numerical label into alphabetical, e.g. 4 = 'posterior/tDCS'
        
        if choose_label and choose_label != label: continue
        
        if   len(x.shape) == 3:  x = np.expand_dims(x, axis=0)
        elif len(x.shape) != 4:  raise Exception(f"The 'x' values should have 4 dimensions, but damn I got one with {len(x.shape)} dimensions instead!")
        
        y = int(y)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            grid = gradCAM.explain(validation_data=(x, y),
                                   model=model,
                                   class_index=y,                                   #class_index value should match the validation_data
                                   colormap=cmapy.cmap('gray'),                     #use grayscale for straightforward interpretation of feature importance
                                   image_weight=0)                                  #0 = activations map only,  >0 = activations map + input image
        
        if not np.all(grid == 0):
            explainers[label].append(grid)
        
            for colormap in colormaps:
                fig = plt.figure(figsize=(figwidth, 3))
                plt.title(f"ACTIVATIONS — '{colormap}' colormap")
                plt.imshow(grid[:,:,0], cmap=colormap)
                plt.colorbar()
                plt.show()                
                print('\n\n')
    
    explainers = {k: np.array(v) for k,v in explainers.items() if v}                #convert list entries to numpy arrays & remove any empty dictionary entries
    
    return explainers



def visualize_importances(explainers, accuracy, to_evaluate='Holdout', colormaps=['hot'], figwidth=14, save_externally=False):
    mne.set_log_level(verbose='CRITICAL')
    
    bad_channels = ['E67', 'E68', 'E69', 'E73', 'E74', 'E82', 'E83', 'E91', 'E92', 'E93', 'E94', 'E102', 'E103', 'E111', 'E120', 'E133', 'E145', 'E146',
                    'E156', 'E165', 'E174', 'E187', 'E190', 'E191', 'E192', 'E199', 'E200', 'E201', 'E202', 'E208', 'E209', 'E210', 'E216', 'E217', 'E218',
                    'E219', 'E225', 'E226', 'E227', 'E228', 'E229', 'E230', 'E231', 'E232', 'E233', 'E234', 'E235', 'E236', 'E237', 'E238', 'E239', 'E240',
                    'E241', 'E242', 'E243', 'E244', 'E245', 'E246', 'E247', 'E248', 'E249', 'E250', 'E251', 'E252', 'E253', 'E254', 'E255', 'E256', 'E257']
    
    
    feature_importance, channel_importance = {}, {}
    
    ##Relabel features for publication-ready figures (Source -> feature_labels.csv):
    nice_labels = ['Mean (µV)', 'Median (µV)', 'Standard Deviation (µV)', 'Max (µV)', 'Min (µV)', 'Power in Delta band (µV2 / Hz)', 'Power in Theta band (µV2 / Hz)', 
        'Power in Alpha band (µV2 / Hz)', 'Power in Beta band (µV2 / Hz)', 'Power in Gamma band (µV2 / Hz)', 'Power in Delta band (dB)', 'Power in Theta band (dB)', 'Power in Alpha band (dB)', 
        'Power in Beta band (dB)', 'Power in Gamma band (dB)', 'Mode of z-scored distribution (5-bin histogram)', 'Mode of z-scored distribution (10-bin histogram)',
        'Longest period of consecutive values above the mean', 'Time intervals between successive extreme events above the mean', 'Time intervals between successive extreme events below the mean', 
        'First 1 / e crossing of autocorrelation function', 'First minimum of autocorrelation function', 'Total power in lowest fifth of frequencies in the Fourier power spectrum', 
        'Centroid of the Fourier power spectrum', 'Mean error from a rolling 3-sample mean forecasting', 'Time-reversibility statistic, ⟨(xt+1−xt)3⟩t', 'Automutual information, m = 2, τ = 5', 
        'First minimum of the automutual information function', 'Proportion of successive differences exceeding 0.04σ', 'Longest period of successive incremental decreases', 
        'Shannon entropy of two successive letters in equiprobable 3-letter symbolization', 'Change in correlation length after iterative differencing', 'Exponential fit to successive distances in 2-d embedding space', 
        'Proportion of slower timescale fluctuations that scale with DFA (50% sampling)', 'Proportion of slower timescale fluctuations that scale with linearly rescaled range fits',
        'Trace of covariance of transition matrix between symbols in 3-letter alphabet', 'Periodicity measure of (Wang et al. 2007)']
    
    if   metadata['DATA_TYPE'] == 'timeseries': nice_labels = list(range(metadata['EEG_RESAMPLING_RATE']))
    elif metadata['DATA_TYPE'] == 'features':   nice_labels = [label + f' [F{i}]' for i, label in enumerate(nice_labels)]
        
    
    for key, activations in explainers.items():
        ##Compute the feature-wise & channel-wise normalized average activations (to be used as proxies for importance):
        assert len(activations.shape) == 4
        mean_batches  = np.mean(activations, axis=0)                    #collapse activations.shape=(batches, features, EEG_ch, 3) accross batches (>=1)
        
        mean_features = np.mean(mean_batches, axis=(1, 2))              #collapse mean_batches.shape=(features, EEG_ch, 3) accross EEG_ch (~188) & RGB_ch (3)
        mean_channels = np.mean(mean_batches, axis=(0, 2))              #collapse mean_batches.shape=(features, EEG_ch, 3) accross features (37) & RGB_ch (3)
        
        norm_mean_batches =  (mean_batches  - mean_batches.min())  / (mean_batches.max()  - mean_batches.min())                 #MinMax normalization on averaged batches
        norm_mean_features = (mean_features - mean_features.min()) / (mean_features.max() - mean_features.min())                #MinMax normalization on feature importances
        norm_mean_channels = (mean_channels - mean_channels.min()) / (mean_channels.max() - mean_channels.min())                #MinMax normalization on channel importances
        
        unsorted_f = {k: v for k,v in zip(nice_labels, norm_mean_features)}                                                     #match feature importances with corresponding labels
        feature_importance[key] = {i[0]: i[1] for i in sorted(unsorted_f.items(), key=operator.itemgetter(1), reverse=True)}    #sort in descending order
        
        unsorted_c = {k: v for k,v in zip(metadata['channel_labels'], norm_mean_channels)}                                      #match channel importances with corresponding labels
        channel_importance[key] = {i[0]: i[1] for i in sorted(unsorted_c.items(), key=operator.itemgetter(1), reverse=True)}    #sort in descending order
        
        
        ##Visualize model explainers as average Activation Maps:
        for colormap in colormaps:
            fig = plt.figure(figsize=(figwidth, 9))
            ax = fig.add_subplot(111)
            plt.title(f"'{colormap}' Activations Map for '{key}' classification\n", fontstyle='italic')
            plt.imshow(norm_mean_batches[:,:,0], cmap=colormap)
            plt.xlabel('\nElectrode ID')
            plt.ylabel('Feature', fontvariant='small-caps')
            plt.xticks(range(0, len(metadata['channel_labels']), 5), metadata['channel_labels'][::5], rotation=90)
            plt.yticks(range(len(nice_labels)), nice_labels)
            plt.colorbar(fraction=0.0275, pad=0.04)
            ax.set_aspect(3)
            plt.show()
            
            if save_externally: gradCAM.save(norm_mean_batches, output_dir=".", output_name=f"GradCAM-{key}-{colormap}.png")
            print('\n\n')
        
        
        ##Visualize Feature Importances as horizontal bar plots:
        fig = plt.figure(figsize=(figwidth, 8))
        plt.title(f"Feature Importance for '{key}' class —  {np.round(accuracy*100, 1)}% {to_evaluate} accuracy")
        plt.barh(list(feature_importance[key].keys()), 
                 list(feature_importance[key].values()),
                 color='red')
        plt.gca().invert_yaxis()
        plt.xlabel('\nNormalized Activation')
        plt.ylabel('Feature\n')
        plt.show()
        
        print('\n\n')
        
        
        ##Visualize channel importances, as color-grouped vertical bar plots:
        n_max = 70

        fig = plt.figure(figsize=(figwidth, 5))
        plt.title(f"Channel Importance for '{key}' class ({n_max} top electrodes)  —  {np.round(accuracy*100, 1)}% {to_evaluate} accuracy")
        plt.bar(list(channel_importance[key].keys())[:n_max], 
                list(channel_importance[key].values())[:n_max], 
                color='limegreen')
        plt.xlabel('\nNormalized Activation')
        plt.ylabel('Channel\n')
        plt.xticks(rotation=90)
        plt.show()

        print('\n\n')
        
        
        ##Visualize Channel Importances as colored topomaps of electrodes:
        title = f"EEG channels importance for '{key}'  — {accuracy} {to_evaluate} accuracy\n"
        
        egi_montage = mne.channels.make_standard_montage('GSN-HydroCel-257')

        ch_names = [c for c in egi_montage.ch_names if c not in bad_channels and c!='Cz']
        
        simulated_info = mne.create_info(ch_names=ch_names, sfreq=250., ch_types='eeg')
        
        scaled_importance = np.array(list(unsorted_c.values())) - 0.5                  #scale normalized values to +- 0.5 range
        scaled_importance = np.expand_dims(scaled_importance, axis=-1)                              #necessary to create an EvokedArray object 

        simulated_evoked = mne.EvokedArray(scaled_importance, simulated_info)        
        simulated_evoked.set_montage(egi_montage)
            
        plt.title(f"Topomap of Channel Importance — {np.round(accuracy*100, 1)}% {to_evaluate} '{key}' accuracy")
        mne.viz.plot_topomap(simulated_evoked.data[:, 0], simulated_evoked.info)
        
        
        print(f"\n\n\n{'='*120}\n\n\n")
    
    
    return feature_importance, channel_importance



def evaluate_TF_model(model, dataset, tf_history=None, to_evaluate='Holdout', load_previous=None, version=-1, VERBOSE=True):    
    ##Plot evolution of training Accuracy & Loss:    
    if tf_history:
        plt.plot(tf_history.history['accuracy']    , label = 'accuracy')
        plt.plot(tf_history.history['val_accuracy'], label = 'val_accuracy')

        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Inter-subject Classification ({} participants, {} Train/Val task)'.format(len(metadata['participant_IDs']), metadata['DATA_TYPE']))
        # plt.ylim([0.5, 1])
        plt.xticks(range(len(tf_history.history['accuracy'])))
        plt.legend(loc='lower right')
        plt.show()
        
        print('\n\n')
        
        plt.plot(tf_history.history['loss']    , label = 'loss')
        plt.plot(tf_history.history['val_loss'], label = 'val_loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Inter-subject Loss ({} participants, {} Train/Val task)'.format(len(metadata['participant_IDs']), metadata['DATA_TYPE']))
        # plt.ylim([0, 2])
        plt.xticks(range(len(tf_history.history['val_accuracy'])))
        plt.legend(loc='upper right')
        plt.show()
        
        print('\n\n')
    
    
    ##Evaluate performance of our model, and save to external file if good enough:
    if load_previous: model = load_TF_model(load_previous='saved_model', version=-1, VERBOSE=VERBOSE)    
    if type(dataset) == dict: dataset = dataset[to_evaluate]
    
    
    loss, accuracy = model.evaluate(dataset, verbose=VERBOSE)

    if VERBOSE: print(f"\n{to_evaluate} Loss: {np.round(loss, 3)}, {to_evaluate} Accuracy: {np.round(accuracy, 3)}\n")

    if tf_history and to_evaluate=='Holdout' and accuracy > 0.6:
        model.save("support-data/DL/model_saved/{}/holdout_acc{}-{}".format(metadata['DATA_TYPE'], np.round(accuracy, 5), metadata['VERSION_NAME']))    
    
    
    ##Produce a Confusion Matrix & a Classification Report:
    labels = np.concatenate([y for _, y in dataset], axis=0)
    predictions = np.array(list(map(lambda class_preds: np.argmax(class_preds), model.predict(dataset))))

    confusion_matrix = tf.math.confusion_matrix(labels, predictions)

    plt.figure(figsize=(8, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='g', cbar=False, cmap='Blues')
    plt.xticks(np.arange(len(metadata['LABEL_MAP'])) + 0.5, metadata['LABEL_MAP'].keys())
    plt.yticks(np.arange(len(metadata['LABEL_MAP'])) + 0.5, metadata['LABEL_MAP'].keys())
    plt.xlabel("\nPredicted Values")
    plt.ylabel("Real Values\n")
    plt.title("Confusion Matrix — {}% {} accuracy, {} task".format(np.round(accuracy*100, 1), to_evaluate, metadata['DATA_TYPE']))
    plt.show()
    
    
    # label_names = [k for k,v in metadata['LABEL_MAP'].items() if v in labels]               #translate numerical label into alphabetical, e.g. 4 = 'posterior/tDCS'
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        report = classification_report(labels, predictions, target_names=list(metadata['LABEL_MAP'].keys()))
    
    print(f"\n\n\n\nClassification Report ({to_evaluate} Dataset):\n{'_'*57}\n\n", report)
    
    
    return loss, accuracy
