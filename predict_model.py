import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.models import load_model
import keras


#---------------------------------------------------------------------------------------------------
file = 'dataset_full.csv'
ni = 144*0
nf = 144*247
df = pd.read_csv(file, header=0)

time = np.array(df['time_index'],dtype=float)[ni:nf]



imag3a = np.array(df['imag3a'],dtype=float)[ni:nf]
iph3a = -180.0 + np.array(df['iph3a'],dtype=float)[ni:nf]
imag3b = np.array(df['imag3b'],dtype=float)[ni:nf]
iph3b = -180.0 + np.array(df['iph3b'],dtype=float)[ni:nf]
imag3c = np.array(df['imag3c'],dtype=float)[ni:nf]
iph3c = -180.0 + np.array(df['iph3c'],dtype=float)[ni:nf]
imag3n = np.array(df['imag3n'],dtype=float)[ni:nf]
iph3n = np.array(df['iph3n'],dtype=float)[ni:nf]


#Build the entire time series
MTS_imag = np.stack([imag3a, imag3b, imag3c, imag3n], axis=1)


#---------------------------------------------------------------------------------------------------
# Settings
#---------------------------------------------------------------------------------------------------
global_settings =  {
                        'training_split':               0.6,
                        'validation_split':             0.2,
                        'shuffle_random_state':         3,
                        'example':                      5, #2,3,5,9,12,18 para el H3
                        'time_steps_offset':            68,
                        'features':                     [0, 1, 2, 3],
                        'best_model_json_imag':         './model_BiLSTM_M3.json',
                        'best_model_weights_imag':      './model_BiLSTM_M3.h5',
                        'best_model_json_iph':          './model_BiLSTM_P3.json',
                        'best_model_weights_iph':       './model_BiLSTM_P3.h5'
                    }

supervised_transform = {
                            'n_in':  144,
                            'n_out': 144,
                            'n_out_features': 4

                        }

figure =    {
                'fname':                './Fig3_results_H3_Day.svg',
                'dpi':                  600,
                'figsize':              [8.75, 15],
                'fontname':             'Times New Roman',
                'alpha':                0.6,
                'sharex':               True,
                'left':                 0.16,
                'right':                0.838,
                'top':                  0.98,
                'bottom':               0.07,
                'wspace':               0.04,
                'hspace':               0.2,
                'show':                 True,
                'save':                 True,
                'interval_colors':      ['g', 'm'],
                'xlims':                [0,144],
                'xticks_frequency':     6*4,
                'xlabels_frequency':    4,
                'xlabel':               'Time (Hours)',
                'xlabelpad':            2,
                'y1lims':               [[2.75,4.25],
                                         [0.75,2.25],
                                         [3.75,5.25],
                                         [7.5,10.5]],
                'y2lims':               [[-120,-90],
                                         [-155,-125],
                                         [-160,-130],
                                         [-150,-120]],
                'y1ticks_frequency':    [0.25,0.25,0.25,0.5],
                'y2ticks_frequency':    [5,5,5,5],
                'y1labels':             ['Current (A)', 'Current (A)', 'Current (A)', 'Current (A)'],
                'y1labelpad':           [3, 3, 3, 3],
                'y2labels':             ['Phase angle (º)', 'Phase angle (º)', 'Phase angle (º)', 'Phase angle (º)'],
                'y2labelpad':           [3, 3, 3, 3],
                'axis_label_fontsize':  10,
                'tick_label_fontsize':  10,
                'y1legend':              [['Ground truth ($I_{A_3}$)', 'BiLSTM'],
                                          ['Ground truth ($I_{B_3}$)', 'BiLSTM'],
                                          ['Ground truth ($I_{C_3}$)', 'BiLSTM'],
                                          ['Ground truth ($I_{N_3}$)', 'BiLSTM']],
                'y2legend':              [['Ground truth ($\\phi_{A_3}$)', 'BiLSTM'],
                                          ['Ground truth ($\\phi_{B_3}$)', 'BiLSTM'],
                                          ['Ground truth ($\\phi_{C_3}$)', 'BiLSTM'],
                                          ['Ground truth ($\\phi_{N_3}$)', 'BiLSTM']],
                'legend_fontsize':      8,
                'legend_location':      ['upper left', 'lower right'],
                'legend_cols':          [3, 3],
                'legend_transparency':  0.6,
                'plot_linewidth':       1.0
          }


def to_supervised(dataset, n_in, n_out, n_out_features):
    """
    Function to convert a time series into a supervised learning problem with X/Y

    ARGUMENTS:
    dataset:            Time series with the folowing format: [[x0, y0, z0...],[x1, y1, z1...]...]
    n_in:               Input time steps (i.e.: 10, 20, 30...)
    n_out:              Output time steps (i.e.: 5, 10, 15...)
    n_out_features:     Just the first n_out_features input features will be available in Y, therefore,
                        it must be less than or equal to the number of input features (i.e.: 1, 2, 3, 4...).

    RETURNS:
    X/Y:                Input and output data (supervised learning)
    """
    #Transform dataset into a supervised learning problem
    X = []
    Y = []
    #Build a version of the dataset without the features to be removed at the output
    if((len(np.shape(dataset)))<2):                     #Single feature time series
        dataset_without_features = dataset.copy()
    elif(n_out_features>=np.shape(dataset)[1]):         #All features at the output
        dataset_without_features = dataset.copy()
    else:
        dataset_without_features = np.delete(arr=dataset, obj=range(n_out_features,np.shape(dataset)[1]), axis=1)

    if(len(dataset)>n_in+n_out+1):
        for i in range(len(dataset)-n_in-n_out+1):
            X.append(dataset[i:i+n_in])
            Y.append(dataset_without_features[i+n_in:i+n_in+n_out])
        return np.array(X), np.array(Y)
    else:
        return np.array([]), np.array([])
#---------------------------------------------------------------------------------------------------
#Transform all the MTS in a supervised learning problem
#---------------------------------------------------------------------------------------------------
#Time
_, Y_time = to_supervised(dataset=time,
                               n_in=supervised_transform['n_in'],
                               n_out=supervised_transform['n_out'],
                               n_out_features=supervised_transform['n_out_features'])

#Magnitude
X_imag, Y_imag = to_supervised(dataset=MTS_imag,
                               n_in=supervised_transform['n_in'],
                               n_out=supervised_transform['n_out'],
                               n_out_features=supervised_transform['n_out_features'])

order = np.arange(len(X_imag))

#Shuffle samples to improve generalization
X_imag, Y_imag,Y_time, order = shuffle(X_imag, Y_imag,Y_time, order, random_state=global_settings['shuffle_random_state'])

#Get only the test set
#Magnitude
testX_imag = X_imag[int(len(X_imag)*(global_settings['training_split']+global_settings['validation_split'])):]
testY_imag = Y_imag[int(len(Y_imag)*(global_settings['training_split']+global_settings['validation_split'])):]

order_test = order[int(len(Y_imag)*(global_settings['training_split']+global_settings['validation_split'])):]
order_test = order_test + global_settings['time_steps_offset']
testY_time = Y_time[int(len(Y_imag)*(global_settings['training_split']+global_settings['validation_split'])):]

#---------------------------------------------------------------------------------------------------
#Normalize 
#--------------------------------------------------------------------------------------------------
max_imag = np.max(a=MTS_imag, axis=0)
min_imag = np.min(a=MTS_imag, axis=0)
testX_normalized_imag = (testX_imag-min_imag)/(max_imag-min_imag)
testY_normalized_imag = (testY_imag-min_imag)/(max_imag-min_imag)


#---------------------------------------------------------------------------------------------------
#Load the models
#---------------------------------------------------------------------------------------------------
#Magnitude
#Load json and create model
file_model = 'BiLSTM'
Model = f'./Ficheros_entrenamiento/{file_model}.h5'
model_imag = load_model(Model,custom_objects={'mse': keras.losses.MeanSquaredError()})

model_iph = load_model(Model,custom_objects={'mse': keras.losses.MeanSquaredError()})
model_imag.compile(optimizer='adam', loss='mse', metrics=['mae'])
model_iph.compile(optimizer='adam', loss='mse', metrics=['mae'])
#---------------------------------------------------------------------------------------------------
#Perform the prediction and retrieve the original scale
#---------------------------------------------------------------------------------------------------
#Magnitude
testY_hat_normalized_imag = model_imag.predict(x=testX_normalized_imag, verbose=0)
testY_hat_imag = min_imag + testY_hat_normalized_imag*(max_imag-min_imag)
testY_imag = min_imag + testY_normalized_imag*(max_imag-min_imag)



#---------------------------------------------------------------------------------------------------
#Order samples by time step
#---------------------------------------------------------------------------------------------------
testY_imag_ordered = []
testY_hat_imag_ordered = []

for i in range(len(testY_imag)):
    testY_imag_ordered.append(np.roll(a=testY_imag[i],shift=order_test[i], axis=0))
    testY_hat_imag_ordered.append(np.roll(a=testY_hat_imag[i],shift=order_test[i], axis=0))

testY_time_ordered = []
for i in range(len(testY_time)):
    testY_time_ordered.append(np.roll(a=testY_time[i],shift=order_test[i], axis=0))

testY_imag_ordered = np.array(testY_imag_ordered)
testY_hat_imag_ordered = np.array(testY_hat_imag_ordered)
testY_time_ordered = np.array(testY_time_ordered)

errors_imag = testY_imag_ordered-testY_hat_imag_ordered
bounds100_imag = []
bounds99_imag = []
bounds100_iph = []
bounds99_iph = []

#Get the error percentiles along time steps
for i in global_settings['features']:
    bounds100_imag.append(np.percentile(a=errors_imag[:,:,i], q=[5,95], axis=0))

bounds100_imag = np.array(bounds100_imag)


#Plots
plt.rcParams['font.family'] = figure['fontname']
fig, ax = plt.subplots(nrows=len(global_settings['features']),
                       ncols=1,
                       figsize=(figure['figsize'][0]/2.54, figure['figsize'][1]/2.54),
                       sharex=figure['sharex'])

for i in global_settings['features']:
    ax[i].grid(which = 'both')
    y1ticks = np.arange(figure['y1lims'][i][0],
                       figure['y1lims'][i][1] + figure['y1ticks_frequency'][i],
                       figure['y1ticks_frequency'][i])
    ax[i].set_yticks(ticks=y1ticks)
    ax[i].set_ylabel(ylabel=figure['y1labels'][i],
                     fontsize=figure['axis_label_fontsize'],
                     labelpad=figure['y1labelpad'][i])
    ax[i].set_yticklabels(labels=y1ticks,
                          rotation=0,
                          fontsize = figure['tick_label_fontsize'])

    ax[i].set_ylim(figure['y1lims'][i])
    ax[i].set_xlim(figure['xlims'])


    ax_twin = ax[i].twinx()
    ax_twin.set_ylim(figure['y2lims'][i])
    y2ticks = np.arange(figure['y2lims'][i][0],
                       figure['y2lims'][i][1] + figure['y2ticks_frequency'][i],
                       figure['y2ticks_frequency'][i])
    ax_twin.set_yticks(ticks=y2ticks)
    ax_twin.set_ylabel(ylabel=figure['y2labels'][i],
                       fontsize=figure['axis_label_fontsize'],
                       labelpad=figure['y2labelpad'][i])
    ax_twin.set_yticklabels(labels=y2ticks,
                            rotation=0,
                            fontsize = figure['tick_label_fontsize'])
    
    #plt.plot(np.arange(len(bounds[0] + testY_hat[example,:, feature])), bounds[0] + testY_hat[example,:, feature], color=plot['bounds_color'])
    #plt.plot(np.arange(len(bounds[1] + testY_hat[example,:, feature])), bounds[1] + testY_hat[example,:, feature], color=plot['bounds_color'])
    ax[i].plot(np.arange(len(testY_imag_ordered[global_settings['example'],:, i])),
               testY_imag_ordered[global_settings['example'],:, i],
               color='C3',
               label=figure['y1legend'][i][0],
               linewidth=figure['plot_linewidth'])
    
    #ax[i].plot(np.arange(len(testY_hat_imag[global_settings['example'],:, i])),
    #           testY_hat_imag[global_settings['example'],:, i] , 
    #           color='k',
    #           label=figure['legend'][0][1],
    #           linestyle='dashdot',
    #           linewidth=figure['plot_linewidth'])
    
    ax[i].fill_between(x=np.arange(len(bounds100_imag[i][0])),
                       y1=bounds100_imag[i][0] + testY_hat_imag_ordered[global_settings['example'],:, i],
                       y2=bounds100_imag[i][1] + testY_hat_imag_ordered[global_settings['example'],:, i],
                       alpha=figure['alpha'],
                       facecolor=figure['interval_colors'][0],
                       label=figure['y2legend'][i][1])

    
    #ax_twin.plot(np.arange(len(testY_hat_iph[global_settings['example'],:, i])),
    #             testY_hat_iph[global_settings['example'],:, i] ,
    #             color='m',
    #             label=figure['legend'][1][1],
    #             linestyle='dashdot',
    #             linewidth=figure['plot_linewidth'])
    
    

    
    ax[i].legend(loc=figure['legend_location'][0],
                 fontsize=figure['legend_fontsize'],
                 ncols=figure['legend_cols'][0],
                 framealpha=figure['legend_transparency'])
    
    ax_twin.legend(loc=figure['legend_location'][1],
                   fontsize=figure['legend_fontsize'],
                   ncols=figure['legend_cols'][1],
                   framealpha=figure['legend_transparency'])







xticks = np.arange(figure['xlims'][0],
                   figure['xlims'][1] + figure['xticks_frequency'],
                   figure['xticks_frequency'])
ax[len(global_settings['features'])-1].set_xticks(ticks=xticks)

ax[len(global_settings['features'])-1].set_xticklabels(labels=np.arange(0,figure['xlabels_frequency']*len(xticks), figure['xlabels_frequency']),
                                                       rotation=0,
                                                       fontsize = figure['tick_label_fontsize'])
ax[len(global_settings['features'])-1].set_xlabel(xlabel=figure['xlabel'],
                                             fontsize=figure['axis_label_fontsize'],
                                             labelpad=figure['xlabelpad'])
plt.subplots_adjust(left=figure['left'],
                    bottom=figure['bottom'],
                    right=figure['right'],
                    top=figure['top'],
                    wspace=figure['wspace'],
                    hspace=figure['hspace'])

if(figure['show']):
    plt.show()
if(figure['save']):
    fig.savefig(fname=figure['fname'], dpi=figure['dpi'])
