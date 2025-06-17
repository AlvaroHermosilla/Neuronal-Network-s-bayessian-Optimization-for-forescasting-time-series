# Importamos los modulos.
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import datetime
import time
import keras_tuner as kt
from tensorflow.keras.layers import Conv1D,GRU,LSTM, Dropout, Dense, TimeDistributed
from tensorflow.keras.models import Model
#Agregado
import gc
from keras import ModelCheckpoint
from keras import backend as K
from tensorflow.keras import mixed_precision
import json

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# 1 - Importación de los datos
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# 1.1 - Importamos los datos del Dataset.
ni = 144*0
nf = 144*247
df = pd.read_csv('./Carga_modelo/dataset_full.csv', header=0)
#1.2 - Convertimos a array las columnas del tercer armonico y quinto armonico.
imag3a = np.array(df['imag3a'],dtype=float)[ni:nf]
iph3a = -180.0 + np.array(df['iph3a'],dtype=float)[ni:nf]
imag3b = np.array(df['imag3b'],dtype=float)[ni:nf]
iph3b = -180.0 + np.array(df['iph3b'],dtype=float)[ni:nf]
imag3c = np.array(df['imag3c'],dtype=float)[ni:nf]
iph3c = -180.0 + np.array(df['iph3c'],dtype=float)[ni:nf]
imag3n = np.array(df['imag3n'],dtype=float)[ni:nf]
iph3n = np.array(df['iph3n'],dtype=float)[ni:nf]


# 1.3 - Por ultimo formamos la serie temporal multivariada con los datos del tercer y quinto armonico.
MTS = np.stack([imag3a,imag3b,imag3c,imag3n], axis=1)



#1.4 - Diccionarios necesarios para las funciones.

supervised_transform = {
                            'n_in':            144,
                            'n_out':           144,
                            'n_out_features':  4
                        }
global_settings =   {
                        'training_split':       0.4,
                        'validation_split':     0.8
                    }
Hyperparameter = {
                            'min_layer': 1,                                                            
                            'max_layer': 5,                                                            
                            'min_neurons': 32,                                                         
                            'max_neurons': 128,
                            'kernel_min' : 2,
                            'kernel_max' : 7,                                                    
                            'activation_functions':['relu','tanh','sigmoid'],                          
                            'lr_min':0.01,
                            'lr_max':0.1,                                          
                            }
Parameters =    {   
                    'verbose':              1,
                    'Dropout':              0.1,
                    'epochs_max':           300,
                    'warmup_epochs':        30,
                    'lr_max':               0.1,
                    'batch_size':           512,
                    'loss':                 'mse'
                }
Tuner = {
            'objective': 'val_loss',
            'max_trials': 40,
            'directory': "afinacion_red",
            'project_name': 'Tuner',
            'overwrite': True,
            'num_initial': 15,
    
}

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# 2 - Preprocesamiento de los datos
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# 2.1  - Convertimos la MTS   (A la hora de tomar los datos deberemos de segguir una metodologia: autocorrelacion(ACF) y autocorrelacion parcial(PACF))(mirar)
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
def clear_memory():
    tf.keras.backend.clear_session()
    gc.collect()
class DenormalizedMSE(tf.keras.metrics.Metric):
    def __init__(self, y_min, y_max, name='denorm_mse', **kwargs):
        super(DenormalizedMSE, self).__init__(name=name, **kwargs)
        self.y_min = tf.constant(y_min, dtype=tf.float32)
        self.y_max = tf.constant(y_max, dtype=tf.float32)
        self.mse = tf.keras.metrics.MeanSquaredError()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        y_true_denorm = y_true * (self.y_max - self.y_min) + self.y_min
        y_pred_denorm = y_pred * (self.y_max - self.y_min) + self.y_min

        self.mse.update_state(y_true_denorm, y_pred_denorm, sample_weight)

    def result(self):
        return self.mse.result()

    def reset_states(self):
        self.mse.reset_states()



X, Y= to_supervised(dataset=MTS,
                    n_in=supervised_transform['n_in'],
                    n_out=supervised_transform['n_out'],
                    n_out_features=supervised_transform['n_out_features'])

X, Y = shuffle(X, Y,random_state=1)  # Mezclamos los datos para evitar sesgos.

X = np.array(X)
Y = np.array(Y)

# 2.2 - Normlizamos los datos 

Xmax = np.max(a=X, axis=0)
Xmin = np.min(a=X, axis=0)
Ymax = np.max(a=Y, axis=0)
Ymin = np.min(a=Y, axis=0)
X_normalized = (X-Xmin)/(Xmax-Xmin)
Y_normalized = (Y-Ymin)/(Ymax-Ymin)

# 2.3 - Dividimos los datos en entrenamiento,validación y test.
X_train = X_normalized[:int(global_settings['training_split']*len(X_normalized))]
X_valid = X_normalized[int(global_settings['training_split']*len(X_normalized)):int(global_settings['validation_split']*len(X_normalized))]
X_test = X_normalized[int(global_settings['validation_split']*len(X_normalized)):]

Y_train = Y_normalized[:int(global_settings['training_split']*len(Y_normalized))]
Y_valid = Y_normalized[int(global_settings['training_split']*len(Y_normalized)):int(global_settings['validation_split']*len(Y_normalized))]
Y_test = Y_normalized[int(global_settings['validation_split']*len(Y_normalized)):]

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# 3 - Definición de los modelos
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
# 3.1 - Modelo MLP ( Multilayer Perceptron)
def model_MLP(hp):
    clear_memory()
    with strategy.scope():
            #Creamos la capa de entrada.
            inputs = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]),name='input layer')
            #Inicializamos la variable x.
            x = inputs       
            num_layers = hp.Int('Num Layers',Hyperparameter['min_layer'],Hyperparameter['max_layer'])           
            #Añadimos las capas ocultas.                                                                                           
            for i in range(num_layers):
                x = tf.keras.layers.TimeDistributed(
                        tf.keras.layers.Dense(
                            units = hp.Int(f'Layer Neurons {i+1}', Hyperparameter['min_neurons'], Hyperparameter['max_neurons']),
                            activation = hp.Choice(f'Activation Function{i+1}', Hyperparameter['activation_functions']),
                    )   )(x)
                x = tf.keras.layers.Dropout(rate=Parameters['Dropout'])(inputs=x)

            #Creamos la capa de salida.
            outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=Y_train.shape[2]))(x)    
            #Tasa de aprendizaje.
            initial_lr = hp.Float('learning_rate', min_value=Hyperparameter['lr_min'], max_value=Hyperparameter['lr_max'])
            #Definimos el modelo
            model = tf.keras.Model(inputs=inputs, outputs=outputs) 

        #Compilamos el modelo.
            model.compile( 
                        loss=Parameters['loss'],
                        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr ,beta_1=0.9,beta_2=0.98,epsilon=1e-8),
                        metrics=[DenormalizedMSE(y_min=Ymin, y_max=Ymax)]
            )
            return model
# 3.2 - Modelos RNN (Recurrent Neural Network) y Bidirectional RNN

def model_LSTM(hp):
    clear_memory()
    with strategy.scope():
            #Creamos la capa de entrada.
            inputs = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]),name='input layer')
            #Inicializamos la variable x.
            x = inputs   
            num_layers = hp.Int('Num Layers',Hyperparameter['min_layer'],Hyperparameter['max_layer'])               
            #Añadimos las capas ocultas.                                                                                           
            for i in range(num_layers):
               x = LSTM(units = hp.Int(f'Layer Neurons {i+1}', Hyperparameter['min_neurons'], Hyperparameter['max_neurons']), 
                                      activation = 'tanh',
                                      recurrent_activation = 'sigmoid', 
                                      return_sequences=True)(x)
               x  = tf.keras.layers.Dropout(rate=Parameters['Dropout'])(inputs=x)

            #Creamos la capa de salida.
            outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=Y_train.shape[2]))(x)    
            #Tasa de aprendizaje.
            initial_lr = hp.Float('learning_rate', min_value=Hyperparameter['lr_min'], max_value=Hyperparameter['lr_max'])
            #Definimos el modelo
            model = tf.keras.Model(inputs=inputs, outputs=outputs) 

        #Compilamos el modelo.
            model.compile( 
                        loss=Parameters['loss'],
                        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr ,beta_1=0.9,beta_2=0.98,epsilon=1e-8),
                        metrics=[DenormalizedMSE(y_min=Ymin, y_max=Ymax)]
            )
            return model

def model_GRU(hp):
    clear_memory()
    with strategy.scope():
            #Creamos la capa de entrada.
            inputs = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]),name='input layer')
            #Inicializamos la variable x.
            x = inputs                  
            #Añadimos las capas ocultas.                                                                                           
            for i in range(hp.Int('Num Layers',Hyperparameter['min_layer'],Hyperparameter['max_layer'])):
               x = GRU(units = hp.Int(f'Layer Neurons {i+1}', Hyperparameter['min_neurons'], Hyperparameter['max_neurons']), 
                                      activation = 'tanh',
                                      recurrent_activation = 'sigmoid', 
                                      return_sequences=True)(x)
               x  = tf.keras.layers.Dropout(rate=Parameters['Dropout'])(inputs=x)

            #Creamos la capa de salida.
            outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=Y_train.shape[2]))(x)    
            #Tasa de aprendizaje.
            initial_lr = hp.Float('learning_rate', min_value=Hyperparameter['lr_min'], max_value=Hyperparameter['lr_max'])
            #Definimos el modelo
            model = tf.keras.Model(inputs=inputs, outputs=outputs) 

        #Compilamos el modelo.
            model.compile( 
                        loss=Parameters['loss'],
                        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr ,beta_1=0.9,beta_2=0.98,epsilon=1e-8),
                        metrics=[DenormalizedMSE(y_min=Ymin, y_max=Ymax)]
            )
            return model
def model_BiLSTM(hp):
    clear_memory()
    with strategy.scope():
            #Creamos el tensor de entrada
            inputs = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]),name='input layer')
            #Inicializamos la variable x.
            x = inputs   
            #Primera capa bidireccional.               
            x = tf.keras.layers.Bidirectional(
                    LSTM(
                    units=hp.Int('Layer Neurons 1', Hyperparameter['min_neurons'], Hyperparameter['max_neurons']), 
                    activation='tanh',
                    recurrent_activation='sigmoid', 
                    return_sequences=True
                    ), merge_mode='concat'
                )(x)
            x  = tf.keras.layers.Dropout(rate=Parameters['Dropout'])(inputs=x)

            #Añadimos las capas ocultas.                                                                                           
            for i in range(hp.Int('Num Layers',Hyperparameter['min_layer'],Hyperparameter['max_layer'])):
                x = tf.keras.layers.Bidirectional(
                                    LSTM(
                                        units=hp.Int(f'Layer Neurons {i+1}', Hyperparameter['min_neurons'], Hyperparameter['max_neurons']), 
                                        activation='tanh',
                                        recurrent_activation='sigmoid', 
                                        return_sequences=True
                                        ), merge_mode='concat'
                                )(x)
                x  = tf.keras.layers.Dropout(rate=Parameters['Dropout'])(inputs=x)
            #Creamos la capa de salida.
            outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=Y_train.shape[2]))(x)    
            #Tasa de aprendizaje.
            initial_lr = hp.Float('learning_rate', min_value=Hyperparameter['lr_min'], max_value=Hyperparameter['lr_max'])
            #Definimos el modelo
            model = tf.keras.Model(inputs=inputs, outputs=outputs) 

        #Compilamos el modelo.
            model.compile( 
                        loss=Parameters['loss'],
                        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr ,beta_1=0.9,beta_2=0.98,epsilon=1e-8),
                        metrics=[DenormalizedMSE(y_min=Ymin, y_max=Ymax)]
            )
            return model
def model_BiGRU(hp):
    clear_memory()
    with strategy.scope():
            #Creamos el tensor de entrada
            inputs = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]),name='input layer')
            #Inicializamos la variable x.
            x = inputs   
            #Primera capa bidireccional.               
            x = tf.keras.layers.Bidirectional(
                    GRU(
                    units=hp.Int('Layer Neurons 1', Hyperparameter['min_neurons'], Hyperparameter['max_neurons']), 
                    activation='tanh',
                    recurrent_activation='sigmoid', 
                    return_sequences=True
                    ), merge_mode='concat'
                )(x)
            x  = tf.keras.layers.Dropout(rate=Parameters['Dropout'])(inputs=x)

            #Añadimos las capas ocultas.                                                                                           
            for i in range(hp.Int('num_layers',Hyperparameter['min_layer'],Hyperparameter['max_layer'])):
                x = tf.keras.layers.Bidirectional(
                                    GRU(
                                        units=hp.Int(f'Layer Neurons {i+1}', Hyperparameter['min_neurons'], Hyperparameter['max_neurons']), 
                                        activation='tanh',
                                        recurrent_activation='sigmoid', 
                                        return_sequences=True
                                        ), merge_mode='concat'
                                )(x)
                x  = tf.keras.layers.Dropout(rate=Parameters['Dropout'])(inputs=x)
            #Creamos la capa de salida.
            outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=Y_train.shape[2]))(x)    
            #Tasa de aprendizaje.
            initial_lr = hp.Float('learning_rate', min_value=Hyperparameter['lr_min'], max_value=Hyperparameter['lr_max']) 
            #Definimos el modelo
            model = tf.keras.Model(inputs=inputs, outputs=outputs) 

        #Compilamos el modelo.
            model.compile( 
                        loss=Parameters['loss'],
                        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr ,beta_1=0.9,beta_2=0.98,epsilon=1e-8),
                        metrics=[DenormalizedMSE(y_min=Ymin, y_max=Ymax)]
            )
            return model

# 3.3 - Modelos CNN (Convolutional Neural Network)
def model_Conv1D(hp):
    clear_memory()
    with strategy.scope():
            #Creamos la capa de entrada.
            inputs = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]),name='input layer')
            #Inicializamos la variable x.
            x = inputs                  
            #Añadimos las capas ocultas.                                                                                           
            for i in range(hp.Int('num_layers',Hyperparameter['min_layer'],Hyperparameter['max_layer'])):
               x = Conv1D(filters = hp.Int(f'Layer Neurons {i+1}', Hyperparameter['min_neurons'], Hyperparameter['max_neurons']), 
                                      kernel_size =hp.Int(f'Kernel Size {i+1}', Hyperparameter['kernel_min'], Hyperparameter['kernel_max']),
                                      padding = 'causal',
                                      activation =hp.Choice(f'Activation Function {i+1}', Hyperparameter['activation_functions']) )(x)
               x  = tf.keras.layers.Dropout(rate=Parameters['Dropout'])(inputs=x)

            #Creamos la capa de salida.
            outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=Y_train.shape[2]))(x)    
            #Tasa de aprendizaje.
            initial_lr = hp.Float('learning_rate', min_value=Hyperparameter['lr_min'], max_value=Hyperparameter['lr_max'])
            #Definimos el modelo
            model = tf.keras.Model(inputs=inputs, outputs=outputs) 

        #Compilamos el modelo.
            model.compile( 
                        loss=Parameters['loss'],
                        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr ,beta_1=0.9,beta_2=0.98,epsilon=1e-8),
                        metrics=[DenormalizedMSE(y_min=Ymin, y_max=Ymax)]
            )
            return model

def model_CNN_LSTM(hp):
    clear_memory()
    with strategy.scope():
        #Capa de entrada
        inputs = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]),name='input layer')
        #Inicializamos la variable x.
        x = inputs
        #Capas ocultas convulucionales.
        for i in range(hp.Int('num_layers',Hyperparameter['min_layer'],Hyperparameter['max_layer'])):
                    x = Conv1D(filters = hp.Int(f'Layer Neurons {i+1}', Hyperparameter['min_neurons'], Hyperparameter['max_neurons']), 
                                            kernel_size =hp.Int(f'Kernel Size {i+1}', Hyperparameter['kernel_min'], Hyperparameter['kernel_max']),
                                            padding = 'causal',
                                            activation =hp.Choice(f'Activation Function {i+1}', Hyperparameter['activation_functions']) )(x)
                    x  = tf.keras.layers.Dropout(rate=Parameters['Dropout'])(inputs=x)
                    
        #Capa oculta LSTM
        for i in range(hp.Int('num_layers',Hyperparameter['min_layer'],Hyperparameter['max_layer'])):
            x = LSTM(units = hp.Int(f'Layer Neurons {i+1}', Hyperparameter['min_neurons'], Hyperparameter['max_neurons']), 
                                                activation = 'tanh',
                                                recurrent_activation = 'sigmoid', 
                                                return_sequences=True)(x)
            x  = tf.keras.layers.Dropout(rate=Parameters['Dropout'])(inputs=x)
        #Capa de salida.
        outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=Y_train.shape[2]))(x)
        #Tasa de aprendizaje.
        initial_lr = hp.Float('learning_rate', min_value=Hyperparameter['lr_min'], max_value=Hyperparameter['lr_max'])
        #Definimos el modelo
        model = tf.keras.Model(inputs=inputs, outputs=outputs) 

        #Compilamos el modelo.
        model.compile( 
                        loss=Parameters['loss'],
                        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr ,beta_1=0.9,beta_2=0.98,epsilon=1e-8),
                        metrics=[DenormalizedMSE(y_min=Ymin, y_max=Ymax)]
            )
        return model
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# 4 - Ajuste dinámico de los hiperparámetros.
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
tf.config.optimizer.set_jit(False)  # Activa la compilación JIT (XLA)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
def Modelo_wrapped(model,train,valid,Parameters,Hyperparameter,Tuner):
  
    tf.random.set_seed(1) #Fijamos la semilla
    #Separamos los datos.
    X_train,Y_train = train 
    X_valid,Y_valid = valid
    # Definimos el modelo.

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(Parameters['batch_size']).prefetch(tf.data.AUTOTUNE)
    valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, Y_valid)).batch(Parameters['batch_size']).prefetch(tf.data.AUTOTUNE)

    #Tasa de aprendizaje
    def lr_scheduler(epoch, lr, lr_max=Parameters['lr_max'], warmup_epochs=Parameters['warmup_epochs']):
                    if epoch < warmup_epochs:
                        return (lr_max/warmup_epochs)*(epoch+1)
                    else:
                        return lr_max/np.sqrt(epoch+1-warmup_epochs)
                        #return lr_max*np.exp(-decay_rate*(epoch-warmup_epochs+1))
    lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule=lr_scheduler, verbose=Parameters['verbose'])
    
    #Creamos el tuner usando BayesianOptimazation
    clear_memory()
    with tf.device('/device:GPU:0'):
        tuner = kt.BayesianOptimization(
            hypermodel=model,     
            objective=kt.Objective("denorm_mse", direction="min"),       
            max_trials=Tuner['max_trials'], 
            num_initial_points=Tuner['num_initial'],  
            executions_per_trial=1,  
            directory=Tuner['directory'],  
            project_name=Tuner['project_name'],      
            overwrite = Tuner['overwrite'],           
        )
        #Iniciamos la busqueda de los hiperparametros
        tuner.search(
            train_dataset,
            epochs=Parameters['epochs_max'],
            batch_size=Parameters['batch_size'],
            validation_data=valid_dataset,
            callbacks=[lr_callback]  
        )
    best_hps = tuner.get_best_hyperparameters()[0]
    best_model = tuner.hypermodel.build(best_hps)
    clear_memory()

    return best_model,best_hps
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# 5 - Entrenamiento de los modelos.
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
models = [model_MLP,model_LSTM,model_GRU,model_BiLSTM,model_BiGRU,model_Conv1D,model_CNN_LSTM]
for Model in models:
    model_checkpoint_callback = ModelCheckpoint(
      filepath=f'{Model.__name__}.h5',
      save_weights_only=False,
      monitor="denorm_mse",
      mode='max',
      save_best_only=True
      )
    # 5.1 - Buscamos el mejor modelo.
    with tf.device('/device:GPU:0'):
        best_model,best_hps = Modelo_wrapped(model=Model,train=(X_train,Y_train),valid=(X_valid,Y_valid),Parameters=Parameters,Hyperparameter=Hyperparameter,Tuner=Tuner)
    # 5.2 - Entrenamos el mejor modelo.
    time0 = time.time()
    X_set = np.concatenate([X_train, X_valid])
    Y_set = np.concatenate([Y_train, Y_valid])
    with tf.device('/device:GPU:0'):
      def lr_scheduler(epoch, lr, lr_max=Parameters['lr_max'], warmup_epochs=Parameters['warmup_epochs']):
                    if epoch < warmup_epochs:
                        return (lr_max/warmup_epochs)*(epoch+1)
                    else:
                        return lr_max/np.sqrt(epoch+1-warmup_epochs)
                        #return lr_max*np.exp(-decay_rate*(epoch-warmup_epochs+1))
      lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule=lr_scheduler, verbose=Parameters['verbose'])
      history = best_model.fit(X_set, Y_set,
                                 epochs=Parameters['epochs_max'],
                                 batch_size=Parameters['batch_size'],
                                 callbacks=[lr_callback,model_checkpoint_callback])
    training_time = time.time() - time0
    with open(f'history_{Model.__name__}.json', 'w') as f:
      json.dump(history.history, f)
    # 5.3 - Guardamos los hiperparámetros y el modelo.
    hyperparam_dict = best_hps.values
    print(hyperparam_dict)
    df_hyperparameters = pd.DataFrame([hyperparam_dict])
    df_hyperparameters.to_csv(f'./hyperparameters_{Model.__name__}.csv',
                               mode= 'w',
                               index=False)
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # 6 - Evaluación del modelo.
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # 6.1 - Predicción del modelo.

    testY_hat = best_model.predict(X_test)

    #6.2 Parametros de evaluación.
    error_metrics = []
    RMSE = np.sqrt(np.mean(np.square(Y_test-testY_hat), axis=(1,2)))
    RSE = np.sqrt(np.sum(np.square(Y_test-testY_hat), axis=(1,2)))/np.sqrt(np.sum(np.square(Y_test-np.mean(Y_test, axis=(1,2), keepdims=True)), axis=(1,2)))
    MAE = np.mean(np.abs(Y_test-testY_hat), axis=(1,2))
    epsilon = 0.0001  
    MAPE = 100.0 * np.mean(np.abs((Y_test - testY_hat) / np.where(Y_test == 0, epsilon, Y_test)), axis=(1,2))
    MBE = np.mean(Y_test-testY_hat, axis=(1,2))
    error_metrics.append([np.mean(RMSE), np.std(RMSE),
                        np.mean(RSE), np.std(RSE),
                        np.mean(MAE), np.std(MAE),
                        np.mean(MAPE), np.std(MAPE),
                        np.mean(MBE), np.std(MBE).
                        training_time])
    error_metrics
    df_error_metrics = pd.DataFrame(error_metrics)
    df_error_metrics.to_csv(path_or_buf=f'./error_metrics_{Model.__name__}.csv',
                            header=['RMSE_mean', 'RMSE_std', 'RSE_mean', 'RSE_std', 'MAE_mean', 'MAE_std','MAPE_mean','MAPE_std', 'MBE_mean', 'MBE_std','Time'],
                            mode='w',
                            index=False)

