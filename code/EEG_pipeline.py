import mne
import catch22
import numpy as np
import pandas as pd
import os, glob
import xml.etree.ElementTree as ET
import ipywidgets as widgets
from IPython.display import display, Markdown, Javascript, HTML
from datetime import datetime

import matplotlib.pyplot as plt
plt.rc('figure', dpi=200)                                     #set this value if high plot resolutions are desired for HTML/PDF exports



class Constants:
    def __init__(self):
        self.parameters = {}
        
        self.parameters['EEG_ROOT_PATH'] = widgets.Text(
            value='data/raw-EEG',
            placeholder='Introduce path to root folder...',
            description='📁 Path to EEG recordings: ',
            disabled=False,
            layout=widgets.Layout(width='50%'),
            style={'description_width': '180px'})
        
        self.parameters['INPUT_ROOT_PATH'] = widgets.Text(
            value='support-data/EEG-TES/experimental-data',
            placeholder='Introduce path to root folder...',
            description='📁 Path to input CSV data: ',
            disabled=False,
            layout=widgets.Layout(width='50%'),
            style={'description_width': '180px'})
        
        self.parameters['OUTPUT_ROOT_PATH'] = widgets.Text(
            value='data/preprocessed-EEG',
            placeholder='Introduce path to root folder...',
            description='📁 Path to output CSV data:',
            disabled=False,
            layout=widgets.Layout(width='50%'),
            style={'description_width': '180px'})

        display(self.parameters['EEG_ROOT_PATH'], self.parameters['INPUT_ROOT_PATH'], self.parameters['OUTPUT_ROOT_PATH'])
        
    
    def EEG_parameters(self):
        self.parameters['EEG_RESAMPLING_RATE'] = widgets.BoundedIntText(
            value=250,
            min=100,
            max=1000,
            step=1,
            description='EEG Resampling Rate (Hz):',
            style={'description_width': '180px'})

        self.parameters['WINDOW_LENGTH'] = widgets.BoundedFloatText(
            value=1,
            min=0.1,
            max=60,
            step=0.1,
            description='Window Length (s):',
            style={'description_width': '180px'})

        self.parameters['WINDOW_OVERLAP_RATIO'] = widgets.BoundedFloatText(
            value=0,
            min=0,
            max=100,
            step=0.1,
            description='Window Overlap (%):',
            style={'description_width': '180px'})

        display(widgets.VBox([self.parameters['EEG_RESAMPLING_RATE'], self.parameters['WINDOW_LENGTH'], self.parameters['WINDOW_OVERLAP_RATIO']]))
        
    
    def data_parameters(self):
        filenames = sorted(glob.glob(f"{self.parameters['EEG_ROOT_PATH'].value}/P???/run*.mff"))
        filenames = [s.replace('\\', '/').split('/')[-2:] for s in filenames]
        filenames = pd.DataFrame([[s[0]] + s[1].split('_') for s in filenames])
        
        runs = [r[3:]  for r in sorted(set(filenames.loc[:,1]))]                                #select the 'runX' column & trim the initial 'run' prefix
        dates = [f"{d[-2:]}/{d[-4:-2]}/{d[:-4]}"  for d in sorted(set(filenames.loc[:,2]))]     #select the 'YYYYMMDD' column & reformat as 'DD/MM/YYYY'
        participants = sorted(set(filenames.loc[:,0]))                                          #select the 'PXXX' column


        self.parameters['LOOP_MODE'] = widgets.ToggleButtons(
            options=['ONE', 'MANY', 'ALL'],
            value='ONE',
            description='Loop Mode:',
            button_style='info',                  # 'success', 'info', 'warning', 'danger' or ''
            tooltips=['Run through only 1 EEG file', 'Run through a set of EEG files', 'Run through all EEG files that can be found'])
        
        self.parameters['N_RUN'] = widgets.SelectMultiple(
            options=runs,
            value=[runs[0]],
            rows=len(runs),
            description='Run(s):')

        self.parameters['N_DATE'] = widgets.SelectMultiple(
            options=dates,
            value=[dates[0]],
            rows=len(runs),
            description='Date(s):')

        self.parameters['N_PARTICIPANT'] = widgets.SelectMultiple(
            options=participants,
            value=[participants[0]],
            rows=len(runs),
            description='Participant(s):')
        
        h_box = widgets.HBox([self.parameters['N_RUN'], self.parameters['N_DATE'], self.parameters['N_PARTICIPANT']])
        
        display(self.parameters['LOOP_MODE'])     
        display(Markdown('> <span style="color:gray"> *Multiple values can be selected with <kbd>shift</kbd> and/or <kbd>ctrl</kbd> (or <kbd>command</kbd>) pressed together with mouse clicks or arrow keys.*'))
        display(h_box)

        def toggle_visibility(change):
            if change['new'] == 'ALL':
                h_box.layout.display = 'none'
            elif change['new'] == 'MANY':
                h_box.layout.display = ''
                self.parameters['N_RUN'].value, self.parameters['N_DATE'].value, self.parameters['N_PARTICIPANT'].value = runs[1:], dates, participants
            else:
                h_box.layout.display = ''
                self.parameters['N_RUN'].value, self.parameters['N_DATE'].value, self.parameters['N_PARTICIPANT'].value = [runs[1]], [dates[0]], [participants[0]]

        self.parameters['LOOP_MODE'].observe(toggle_visibility, names='value')
        
    
    def control_parameters(self):
        self.parameters['IS_VIZ'] = widgets.ToggleButton(
            value=True,
            description='✔ Visualization Mode',
            button_style='success',                  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Enable this setting to have Figures & Plots displayed',
            layout=widgets.Layout(width='200px'))

        self.parameters['IS_VERBOSE'] = widgets.ToggleButton(
            value=True,
            description='✔ Verbose Mode',
            button_style='warning',                  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Enable this setting to have functions print what they are doing at each step',
            layout=widgets.Layout(width='200px'))        
        
        self.parameters['EXPORT_MODE'] = widgets.ToggleButtons(
            options=['NONE', 'TIMESERIES_ONLY', 'FEATURES_ONLY', 'ALL'],
            value='ALL',
            description='Data Export Mode:',
            button_style='danger',                   # 'success', 'info', 'warning', 'danger' or ''
            tooltips=['Do not export any CSV files', 'Export only CSVs with the EEG Timeseries', 'Export only CSVs with the EEG Features', 'Export all CSVs'])

        display(widgets.HBox([self.parameters['IS_VIZ'], self.parameters['IS_VERBOSE']], layout=widgets.Layout(width='410px', justify_content='space-between')))
        display(self.parameters['EXPORT_MODE'])
        
        
        def toggle_button(change):
            key = 'IS_VIZ' if 'Visualization' in change['owner'].description else 'IS_VERBOSE'
            globals().update({key: change['new']})                                     #update global value dynamically on button press
            
            if change['new']:
                change['owner'].description = '✔ ' + change['owner'].description      #animate a check mark on button press
            else:
                change['owner'].description = change['owner'].description[2:]

        self.parameters['IS_VIZ'].observe(toggle_button, names='value')
        self.parameters['IS_VERBOSE'].observe(toggle_button, names='value')


class Session:
    def __init__(self, constants, skip_calibration=True, start_from_run=0):
        ##Reformat constants from Widgets so that they make sense to the rest of the pipeline:
        self.constants = {}
        
        self.constants['WINDOW_SIZE']    = int(constants['WINDOW_LENGTH'].value * constants['EEG_RESAMPLING_RATE'].value)    #in samples
        self.constants['WINDOW_OVERLAP'] = constants['WINDOW_LENGTH'].value * constants['WINDOW_OVERLAP_RATIO'].value/100    #in seconds
        
        for k, v in constants.items():
            v = v.value
            
            if type(v) == tuple: v = list(v)                                #cast tuples to lists
                
            if k == 'N_RUN'    : v = [int(i) for i in v]                    #cast N_RUN to integer values
            elif k == 'N_DATE' : v = [d[-4:] + d[3:5] + d[:2] for d in v]   #reformat N_DATE as "YYYYMMDD"

            self.constants[k] = v
        
        globals().update(self.constants)                                    #make constants from Widgets accessible as global variables
        
        ##Define the EEG files to read, according to the 'Loop Mode' widget output:
        if LOOP_MODE == 'ALL':
            filenames = sorted(glob.glob(f"{EEG_ROOT_PATH}/P*/run*_??????.mff"))
            
        elif LOOP_MODE == 'ONE' or LOOP_MODE == 'MANY':
            filenames = []
            for r in N_RUN:
                for d in N_DATE:
                    for p in N_PARTICIPANT:
                        search = glob.glob(f"{EEG_ROOT_PATH}/{p}*/run{r}_{d}_??????.mff")
                        if search != []: filenames += search
            filenames = sorted(filenames)
        
        values = [s.replace('\\', '/').split('/')[-2:] for s in filenames]
        values = [[s[0]] + s[1].split('_') for s in values]
        if skip_calibration: values = [s for s in values if s[1] != 'run0']
            
        self.values = values
        
        self.history = []  if start_from_run==0  else ['skipped #' + str(n) for n in [*range(start_from_run)]]


    
    def begin_loop(self):        
        global N_RUN, N_DATE, N_PARTICIPANT
        if isinstance(N_RUN, list): display(Markdown("> <span style=\"color:gray\">*NOTE: this loop works only in Jupyter Notebooks; if in Jupyter Lab, click 'Help > Launch Classic Notebook'*"))
        
        try:
            values = self.values[len(self.history)]
            N_RUN, N_DATE, N_PARTICIPANT = int(values[1][3:]), values[2], values[0]
        except IndexError:
            print("\n✔️ Alright, session finished!")
            return
        
        if 'google.colab' in str(get_ipython()):
            print("\n⚠️ Can't automatically run the loop in Google Colab. Try in a Jupyter Notebook, or manually run the cells below!")
            return
        else:
            print(f"📢 Iterating through the following {len(self.values)} EEG files:\n")
            for i, file in enumerate(self.values):
                prefix = f'{i}:'  if i != len(self.history)  else f'👉 {i}:'
                print('{:<4s}'.format(prefix), file)
        
        if IS_VIZ:
            print("\n⚠️ Manual execution is preferred when Visualization Mode is enabled. Please disable if you're looking for automated execution!")
        else:
            return Javascript("Jupyter.notebook.execute_cells_below()")

    
    def restart_loop(self, history):
        self.history.append(history)
        
        if not IS_VIZ: return Javascript("""
        var i;
        for (i = 0; i < 30; i++) {
            if (Jupyter.notebook.get_cell(i).get_text().startsWith("session.begin_loop()")){
                Jupyter.notebook.execute_cells([i])
                i = 9999;        
            }
        }
        """)


def load_EEG():
    for file in glob.glob(f"{EEG_ROOT_PATH}/*P*/*run*/signal2.bin"):   #we need to ignore all "signal2.bin" files to avoid errors later with 'mne.io.read_raw_egi' function
        os.rename(file, file + "IGNORE")
        
    metadata = {}   #we will use this dictionary to store all auxiliary data
    metadata['raw_data_path'] = glob.glob(f"{EEG_ROOT_PATH}/{N_PARTICIPANT}/run{N_RUN}_{N_DATE}*.mff")[0].replace('\\', '/')
    
    raw = mne.io.read_raw_egi(metadata['raw_data_path'], preload=True, verbose=IS_VERBOSE)
    if IS_VERBOSE: print(f"\n📢 The EEG file looks like this:\n\n{raw}\n\n{raw.info}\n\nEGI stimulation channel: {raw.copy().pick_types(stim=True).ch_names[0]}\n")
    
    events, metadata = extract_EGI_events(raw, metadata)
    
    annotations, metadata = annotate_EEG(metadata)
    raw.set_annotations(annotations)
    
    return raw, events, metadata


def extract_EGI_events(raw, metadata):
    xml_start = ET.parse(metadata['raw_data_path'] + '/Events_GTEN Injection Block Start.xml')  #read the (untangled) Events from both '.xml' files...
    xml_end   = ET.parse(metadata['raw_data_path'] + '/Events_GTEN Injection Block End.xml')    #...although they come as timestamps, so we'll have to translate into relative time
    
    events_raw = mne.find_events(raw, stim_channel=raw.copy().pick_types(stim=True).ch_names[0], verbose=IS_VERBOSE)
    if IS_VERBOSE: print(f"👁‍🗨👎 Events as provided by EGI in the wrong format:\n{events_raw[:5]}\n ...\n")
    
    events_start = []
    events_end = []

    for tag in xml_start.getroot().findall('.//{http://www.egi.com/event_mff}event'):
        events_start.append(datetime.strptime(tag[0].text, '%Y-%m-%dT%H:%M:%S.%f%z').timestamp())

    for tag in xml_end.getroot().findall('.//{http://www.egi.com/event_mff}event'):
        events_end.append(datetime.strptime(tag[0].text, '%Y-%m-%dT%H:%M:%S.%f%z').timestamp())
    
    events_start = [int(np.round((t-events_start[0]) * 1000 + events_raw[0][0])) for t in events_start]
    events_end   = [int(np.round((t-  events_end[0]) * 1000 + events_raw[1][0])) for t in events_end]

    events_duration = [(end-start)/1000 for end, start in zip(events_end, events_start)]

    for i, duration in enumerate(events_duration):
        if duration < 2.5:
            del events_start[i], events_end[i], events_duration[i]   #sometimes Events come with contaminants of short duration (typically <1s), we need to eliminate them 
    
    labels = pd.read_csv(f'{INPUT_ROOT_PATH}/raw-GTEN-plan.csv', index_col=0).iloc[N_RUN, :]
    labels = [s.strip() for s in labels.values[labels.notnull()].tolist()]
    
    events_df = list(zip(labels, events_start, events_end, [int(np.round(i)) for i in events_duration]))
    events_df = pd.DataFrame(events_df, columns=['label', 'start time', 'end time', 'duration'])
    
    event_data = pd.read_csv(f'{INPUT_ROOT_PATH}/raw-events-data.csv', index_col=0)
    
    ##Format our pandas dataframe as the expected 'Events' object for the MNE pipeline:
    ev_dict = event_data.to_dict()['id']    
    for i, value in enumerate(labels):
        labels[i] = ev_dict[value.strip()]

    events = np.column_stack([events_start, [0] * len(labels), labels])
    if IS_VERBOSE: print(f"👁‍🗨👍 Events in the correct format for the MNE pipeline:\n{events[:5]}\n ...\n")
    
    metadata['events_df'], metadata['event_data'] = events_df, event_data
    
    return events, metadata


def annotate_EEG(metadata):
    epoch_labels = pd.read_csv(f'{INPUT_ROOT_PATH}/epochs-GTEN-plan.csv', index_col=0).iloc[N_RUN, :]
    epoch_labels = [s.strip() for s in epoch_labels.values[epoch_labels.notnull()].tolist()]
    
    epochs_df = pd.DataFrame(epoch_labels, columns=['epoch'])
    epochs_df[['start time', 'end time']] = metadata['events_df'][['start time', 'end time']].div(1000).round(1)
    epochs_df['duration'] = metadata['events_df']['duration']
    
    metadata['tmin'] = 2.5     #in seconds
    metadata['tmax'] = 3.0     #in seconds

    with pd.option_context('mode.chained_assignment', None):                    #this line is only to suppress an unhelpful "SettingWithCopyWarning" message
        for i, row in epochs_df.iterrows():
            if row['epoch'] == 'bad/ramp':
                epochs_df['start time'].iloc[i] = row['start time'] - metadata['tmin']
                epochs_df['duration'].iloc[i]   = row['duration']   + metadata['tmin']
            elif row['epoch'].startswith('bad'):
                epochs_df['end time'].iloc[i]   = row['end time']   + metadata['tmax']
                epochs_df['duration'].iloc[i]   = row['duration']   + metadata['tmax']
    

    annotations = mne.Annotations(onset = epochs_df['start time'].tolist(),     # in seconds
                               duration = epochs_df['duration'].tolist(),       # in seconds
                            description = epochs_df['epoch'].tolist())
                            
    return annotations, metadata


def preprocess_EEG(raw, highpass=1, notch=50, lowpass=100, do_filter=True, mark_channels='bad'):
    if raw.info['lowpass'] < EEG_RESAMPLING_RATE/2:
        print("⚠️ Stopped Execution: seems like the EEG signal was already filtered below the Nyquist frequency!")
        return
    
    ##Remove bad channels. NOTE: this list comes from manually selecting the outermost electrodes (i.e. around eyes, ears & neck), which typically showed bad contact with scalp 
    raw.info['bads'] = ['E67', 'E68', 'E69', 'E73', 'E74', 'E82', 'E83', 'E91', 'E92', 'E93', 'E94', 'E102', 'E103', 'E111', 'E120', 'E133', 'E145', 'E146',
                        'E156', 'E165', 'E174', 'E187', 'E190', 'E191', 'E192', 'E199', 'E200', 'E201', 'E202', 'E208', 'E209', 'E210', 'E216', 'E217', 'E218',
                        'E219', 'E225', 'E226', 'E227', 'E228', 'E229', 'E230', 'E231', 'E232', 'E233', 'E234', 'E235', 'E236', 'E237', 'E238', 'E239', 'E240',
                        'E241', 'E242', 'E243', 'E244', 'E245', 'E246', 'E247', 'E248', 'E249', 'E250', 'E251', 'E252', 'E253', 'E254', 'E255', 'E256', 'E257']
    
    if IS_VIZ: raw.plot_sensors(kind='select', show_names=False)
    if IS_VIZ: raw.plot_sensors(kind='3d', show_names=False)
        
    # _, ch_select[mark_channels] = raw.plot_sensors(kind='select', block=True)                                 #use this line if we wanted to mark good/bad channels interactively
    # raw.info['bads'] = [ch for ch in raw.ch_names if ch not in ch_select['good'] or ch in ch_select['bad]]    #if using the previous line, this one may be handy too (modify as needed)
    
    orig_ch = raw.ch_names
    raw.pick_types(eeg=True, stim=False, exclude='bads')    #permanently remove bad & stim channels
    if IS_VERBOSE: print(f"📢 Dropped {len(orig_ch) - len(raw.ch_names)} bad/stim channels:\n{sorted(set(orig_ch) - set(raw.ch_names))}\n\n")
    
    
    ##Filter our data:
    if do_filter:
        raw.load_data()
        raw.filter(highpass, lowpass, verbose=IS_VERBOSE)
        raw.notch_filter(np.arange(notch, notch*4+1, notch), verbose=IS_VERBOSE)    #remove 50Hz power grid artifact + harmonics
    
    # raw, events = raw.resample(EEG_RESAMPLING_RATE, events=events, verbose=IS_VERBOSE)   #it's preferred to resample with 'decim' instead, in 'mne.Epochs' constructor later on
    
    return raw


def epoch_EEG(raw, events, metadata):
    ##For the Epochs we'd like characterize the Rest events according to their previous stimulation, so we need to label the Events differently:
    epoch_events = events.copy()

    for i, e in enumerate(events):
        e_id = None        
        if e[2] in [2, 3, 5, 98, 99]:
            e_id = e[2] + 100
        elif e[2] in [10, 20, 30, 40]:
            e_id = e[2]/10 + 105
        elif e[2] == 1:                                  #for Rest blocks, we need to apply extra logic
            if N_RUN == 0:                               #different for calibration runs
                e_id = 101 if i<4 else 104
            elif events[i-1][2] in [10, 20, 30, 40]:     #if the previous block was a Stim
                e_id = events[i-1][2] + 100
            elif events[i-2][2] in [10, 20, 30, 40]:     #if the previous block was a Rest, and the one before was a Stim
                e_id = events[i-2][2] + 100 + 1
        
        epoch_events[i] = [e[0], e[1], int(e_id)]

    epoch_events = np.vstack((np.array([0, 0, 98]), epoch_events, np.array([metadata['events_df']['end time'].iloc[-1], 0, 99])))  #include start/end baselines
    
    
    ##Now we will remove segments affected by any neurostimulation, and create short overlapping (or not) windows for our EEG timeseries, based on the newly created Events:
    epoch_events_data = pd.read_csv(f'{INPUT_ROOT_PATH}/epochs-events-data.csv', index_col=0)
    
    relevant_epochs = {key:value for (key,value) in epoch_events_data['id'].to_dict().items() if value in epoch_events[:,2].tolist()}   #filter out missing epoch_event IDs
    
    epoch_events_id    = epoch_events_data.loc[relevant_epochs]['id'].to_dict()
    epoch_events_color = epoch_events_data.loc[relevant_epochs]['color'].to_dict()
    
    window_events = mne.make_fixed_length_events(raw, id=420, start=0, stop=None, duration=WINDOW_LENGTH, overlap=WINDOW_OVERLAP)  #create fixed events every WINDOW_LENGTH seconds
    for i, window in enumerate(window_events):
        window_events[i][2] = epoch_events[np.argmax(epoch_events[:,0] > window[0]) - 1][2]   #label the fixed events according to the segment in which they are found
    
    epochs = mne.Epochs(raw,
                        events=window_events,
                        event_id=epoch_events_id,
                        tmin=0, 
                        tmax=WINDOW_LENGTH - 1/raw.info['sfreq'],
                        reject_by_annotation=True,
                        preload=True,
                        decim=4,                           #downsample by a factor of "decim"; in our case by 4, from 1000Hz to 1000/4 = 250Hz
                        # reject={'eeg': 150e-6},          #reject epochs with peak-to-peak amplitude > 150 µV  (for a more automated approach: http://autoreject.github.io)
                        # flat={'eeg': 1e-6},              #reject epochs with peak-to-peak amplitude < 1 µV    (i.e. epochs containing any flat channel)
                        baseline=None,
                        proj=False,
                        verbose=IS_VERBOSE)
    
    
    ##Independently, it will also be convenient to have the rest blocks as whole segments that can be accessed easily:
    block_duration = metadata['events_df'].loc[metadata['events_df'].label == 'rest'].duration.tolist()[0]
    
    if N_RUN>0: block_events = np.array([ev for ev in epoch_events if ev[-1]==101 or ev[-1]==104 or ev[-1]>=110])
    else:       block_events = np.array([ev for ev in epoch_events if ev[-1]<=104])
    
    block_events_id = {key: ev_id for key, ev_id in epoch_events_id.items() if not key.startswith('bad') and not key.startswith('session')}
    
    whole_blocks = mne.Epochs(raw,
                              events=block_events,
                              event_id=block_events_id,
                              tmin=metadata['tmin'] + 0.8,
                              tmax=block_duration - metadata['tmax'],
                              reject_by_annotation=False,
                              preload=True,
                              decim=4,                     #downsample by a factor of "decim"; in our case by 4, from 1000Hz to 1000/4 = 250Hz
                              # reject={'eeg': 150e-6},    #reject epochs with peak-to-peak amplitude > 150 µV  (for a more automated approach: http://autoreject.github.io)
                              # flat={'eeg': 1e-6},        #reject epochs with peak-to-peak amplitude < 1 µV    (i.e. epochs containing any flat channel)
                              baseline=None,
                              proj=False,
                              verbose=IS_VERBOSE)
    
    metadata['epoch_events'], metadata['epoch_events_id'], metadata['epoch_events_color'] = epoch_events, epoch_events_id, epoch_events_color
    
    relevants = ['frontal', 'posterior'] if N_RUN>0 else ['initial rest', 'eyes closed']
    
    return epochs[relevants], whole_blocks[relevants], metadata


def visualize_Raw_EEG(raw, events, metadata):
    if not IS_VIZ:
        if IS_VERBOSE: print("🦘 Skipping Raw EEG visualization (since IS_VIZ=False)")
        return
        
    relevant_events = {key:value for (key,value) in metadata['event_data'].id.to_dict().items() if value in events[:,2].tolist()}    #filter out missing event IDs
    
    events_id    = metadata['event_data'].loc[relevant_events].id.to_dict()
    events_color = metadata['event_data'].loc[relevant_events].set_index('id').color.to_dict()
    
    mne.viz.plot_raw(raw,
                     events=events,
                     event_id=events_id,
                     color=events_color,
                     duration=30,                    #width  (in seconds)
                     n_channels=20,                  #height
                     remove_dc=False,                #let MNE try to remove/attenuate DC drift (channel-wise)
                     verbose=IS_VERBOSE
                    ).set_size_inches(12, 7)
    
    mne.viz.plot_events(events,
                        event_id=events_id,
                        color=events_color,       #¿should work with 'events_color', but Imperial's RCS Jupyter complains for some unknown reason?
                        sfreq=raw.info['sfreq'], show=False
                       ).set_size_inches(9, 5)
    
    mne.viz.plot_raw_psd(raw,
                         fmin=0,
                         fmax=100,
                         tmin=None,
                         tmax=None
                        ).set_size_inches(12, 5)
    
    return


def visualize_Epoched_EEG(epochs, whole_blocks, metadata):
    if not IS_VIZ:
        if IS_VERBOSE: print("🦘 Skipping Epoched EEG visualization (since IS_VIZ=False)")
        return
    
    ##Visualize both 1-second-long epochs & 20-seconds-long blocks:
    print(f"Plotting a total of {len(epochs)} valid Epochs:\n")
    epochs.plot(n_epochs=10,
                n_channels=15,
                events=metadata['epoch_events'], 
                event_id=metadata['epoch_events_id'],
                event_color=metadata['epoch_events_color'],
#                 group_by='position',
                butterfly=False,
                block=False
               ).set_size_inches(12, 7)
    
    
    print(f"\n\nPlotting a total of 8 non-epoched resting-period Blocks:\n")
    whole_blocks.plot(n_epochs=1,
                     n_channels=15,
                     events=metadata['epoch_events'], 
                     event_id=metadata['epoch_events_id'],
                     event_color=metadata['epoch_events_color'],
#                    group_by='position',
                     butterfly=False,
                     block=False
                    ).set_size_inches(12, 7)
    
    
    print("PSD Plot, averaged for all TES conditions:\n")
    epochs.plot_psd(verbose=IS_VERBOSE).set_size_inches(12, 5)
    
    
    ##Visualize spectrograms of the whole 20-seconds-long rest blocks from each of the 5 conditions:
    spectrograms = {}
    frequencies  = np.arange(1, 20.1, 0.5)
    n_cycles     = 2

    for i, condition in enumerate(['rest', 'frontal/tDCS', 'frontal/tACS', 'posterior/tDCS', 'posterior/tACS']):
        key = condition + '/1' if condition != 'rest' else '2'
        
        if IS_VERBOSE: print(f"{i+1}/6: Computing spectrograms for condition '{condition}' (aka '{key}')")
        
        spectrograms[condition] = mne.time_frequency.tfr_morlet(whole_blocks[key], freqs=frequencies, n_cycles=n_cycles, return_itc=False)

    if IS_VERBOSE: print(f"6/6: Computing 'ALL_CONDITIONS_AVERAGED' spectrograms")
    spectrograms['ALL_CONDITIONS_AVERAGED'] = mne.time_frequency.tfr_morlet(whole_blocks, freqs=frequencies, n_cycles=n_cycles, return_itc=False)
    
    for condition, spectrogram in spectrograms.items():
        spectrogram.plot(combine='mean', show=False, verbose=False)
        
        plt.title(f"'{condition}' Spectrogram (averaged epochs & electrodes)\n\n")
        plt.show()
        
        print('\n\n')
    
    ## Visualize PSD topomaps:
    print("PSD Topomap, averaged for all TES conditions:\n")
    epochs.plot_psd_topomap(verbose=IS_VERBOSE)
    print("PSD Topomap, only for 'rest' conditions, and normalized in value:\n")
    epochs['2'].plot_psd_topomap(normalize=True, verbose=IS_VERBOSE)
    
    return


def compute_EEG_features(unscaled_epochs):
    if IS_VERBOSE: print(f"Calculating features on 'Epochs' object with shape:  (n_epochs, n_channels, window_size) => {np.array(unscaled_epochs).shape}\n")
             
    epochs = unscaled_epochs.get_data() * 1e6       #convert units from Volts (V) to microVolts (µV)
    
    axis = -1                                       #can be 'int' or 'tuple of ints' (useful if we wanted to compute features across the 'channels' axis too)
    features = {}

    features['mean']    = np.mean   (epochs, axis=axis)#, keepdims=True)
    features['median']  = np.median (epochs, axis=axis)
    features['std']     = np.std    (epochs, axis=axis)
    features['max']     = np.max    (epochs, axis=axis)
    features['min']     = np.min    (epochs, axis=axis)
    
    
    ##Now we will calculate band-wise features, keeping them separate for each channel:
    bands_freq = {'delta': np.arange(1, 4),
                  'theta': np.arange(4, 8),
                  'alpha': np.arange(8, 13),
                  'beta' : np.arange(13, 30),
                  'gamma': np.arange(30, 45)}
    
    bands_power, freqs = mne.time_frequency.psd_welch(mne.EpochsArray(epochs, unscaled_epochs.info),
                                                      fmin=bands_freq['delta'][0],
                                                      fmax=bands_freq['gamma'][-1] + 1,
                                                      tmin=None,
                                                      tmax=None,
                                                      n_fft=WINDOW_SIZE,
                                                      average='mean',          #this averages across NFFT bins (not frequencies nor channels), which is irrelevant for our 1-second epochs since n=1 because n_fft=WINDOW_SIZE
                                                      verbose=IS_VERBOSE)
    
    bands_power_dB = 10 * np.log10(bands_power)                                #convert from PSD units (µV2/Hz) to decibels (dB)
    
    if IS_VERBOSE: print(f"\nComputed PSD from {freqs[0]}Hz to {freqs[-1]}Hz, returning a shape of:  (n_epochs, n_channels, freqs) => {bands_power.shape}\n")
    
    for band, freq in bands_freq.items():
        features['power_' + band]    = np.mean(bands_power[:,:,freq], axis)
    for band, freq in bands_freq.items():
        features['dB_power_' + band] = np.mean(bands_power_dB[:,:,freq], axis)
    
    ##Lastly we will calculate the catch22 features (https://github.com/chlubba/catch22):
    try:
        catch22_labels = catch22.catch22_all([0])['names']

        catch22_values = np.array([[catch22.catch22_all(channel)['values'] for channel in epoch] for epoch in epochs])

        for i, label in enumerate(catch22_labels):
            features['catch22-' + label] = catch22_values[:,:,i]
    
    except Exception as e: print("📢 ERROR computing CATCH-22 features (maybe missing 'catch-22' library?) ==>  ", e)
    
    
    if IS_VERBOSE: print(f"Computed {len(features)} features, each with a shape of:  (n_epochs, n_channels) => {features['mean'].shape}\n")
    
    return features
    


def export_CSVs(epochs, features, metadata, do_export=True):
    filename = metadata['raw_data_path'].split('/')[-1][:-4] + '.csv'
    
    resampling_freq = int(epochs.info['sfreq'])
        
    # epochs = epochs.groupby('position')   #this could be useful (also in the 'raw' object) if we wanted the EEG channels ordered by spatial disposition, rather than from 1 to 256
    epochs = epochs.to_data_frame(index='condition').rename_axis('label').drop('time', axis=1)
    
    if EXPORT_MODE=='ALL' or EXPORT_MODE=='TIMESERIES_ONLY':
        timeseries_dir = f'{OUTPUT_ROOT_PATH}/timeseries/{N_PARTICIPANT}/'
        if not os.path.exists(timeseries_dir):  os.makedirs(timeseries_dir)
        
        if IS_VERBOSE:
            print("\nDataFrame containing the epoched EEG timeseries:")
            display(epochs)
            
            print("\n\n\nValue counts of the timeseries labels:")
            display(epochs.index.value_counts())
            
            print(f"\n\n📢 Saving TIMESERIES Epochs to: '{timeseries_dir + filename}'\n")

        if do_export: epochs.to_csv(timeseries_dir + filename)
    
    
    if EXPORT_MODE=='ALL' or EXPORT_MODE=='FEATURES_ONLY':
        features_dir   = f'{OUTPUT_ROOT_PATH}/features/{N_PARTICIPANT}/'
        if not os.path.exists(features_dir):  os.makedirs(features_dir)
        
        features = features.copy()
        
        for key, val in features.items():
            features[key] = pd.DataFrame(val)
            features[key].columns = ['E' + str(col) for col in list(epochs.columns[1:])]
            features[key].insert(loc=0, column='label', value=epochs.index.tolist()[::resampling_freq])
            features[key].insert(loc=0, column='epoch', value=features[key].index)
            features[key].insert(loc=0, column='feature', value=key)

        features = pd.concat(list(features.values())).rename_axis('index').sort_values(['epoch', 'index']).set_index('label')
        
        if IS_VERBOSE:
            print("\nDataFrame containing the extracted EEG features:")
            display(features)
            
            print("\nValue counts of the features labels:")
            display(features.index.value_counts())
            
            print(f"📢 Saving FEATURES Epochs to: '{features_dir + filename}'")

        if do_export: features.to_csv(features_dir + filename)
        
        return features


def update_kaggle_datasets(upload_dataset='all'):
    # os.system("conda env export -p ~/anaconda3/envs/mres > support-data/conda-environment.yml")         #Optionally, take a snapshot of the current anaconda environment
    
    os.environ['KAGGLE_USERNAME'] = "yourUsername"
    os.environ['KAGGLE_KEY'] = "yourKey"
    
    # os.system("kaggle datasets init -p /your/root/folder/data/raw-EEG")             #Only needed once to create 'dataset-metadata.json'
    # os.system("kaggle datasets init -p /your/root/folder/data/preprocessed-EEG")    #Also needed once only
    
    print(f"📢 Commencing upload to Kaggle for '{os.environ['KAGGLE_USERNAME']}' user...")
    
    if upload_dataset=='all' or upload_dataset=='raw':
        os.system("kaggle datasets version -p /your/root/folder/data/raw-EEG -r zip -m 'Updated via Kaggle API for DL-EEG-TES classification'")
        print(f"✔ Uploaded raw EEG data to =>  https://www.kaggle.com/{os.environ['KAGGLE_USERNAME']}/dissertation-raw")
    
    if upload_dataset=='all' or upload_dataset=='preprocessed':
        os.system("kaggle datasets version -p /your/root/folder/data/preprocessed-EEG -r zip -m 'Updated via Kaggle API for DL-EEG-TES classification'")
        print(f"✔ Uploaded preprocessed EEG data to =>  https://www.kaggle.com/{os.environ['KAGGLE_USERNAME']}/dissertation-preprocessed")