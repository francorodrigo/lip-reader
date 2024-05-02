import os 
from keras._tf_keras.keras.models import Sequential 
from keras._tf_keras.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten,TFSMLayer
from keras._tf_keras.keras.models import load_model as load_model2
import h5py as h5py
import tensorflow as tf

def convert_tf1_to_tf2(checkpoint_path, output_prefix):
    """Converts a TF1 checkpoint to TF2.

    To load the converted checkpoint, you must build a dictionary that maps
    variable names to variable objects.
    ```
    ckpt = tf.train.Checkpoint(vars={name: variable})  
    ckpt.restore(converted_ckpt_path)

    ```

    Args:
      checkpoint_path: Path to the TF1 checkpoint.
      output_prefix: Path prefix to the converted checkpoint.

    Returns:
      Path to the converted checkpoint.
    """
    vars = {}
    reader = tf.train.load_checkpoint(checkpoint_path)
    print("reader:",reader)
    dtypes = reader.get_variable_to_dtype_map()
    print("DTYPES", dtypes)
    for key in dtypes.keys():
      vars[key] = tf.Variable(reader.get_tensor(key))
    return tf.train.Checkpoint(vars=vars).save(output_prefix)

def convert_checkpoint_to_h5(checkpoint_path, output_path):
    """
    Convierte un archivo de pesos de Keras antiguo a formato HDF5.

    Args:
        checkpoint_path (str): Ruta del archivo de pesos de Keras antiguo.
        output_path (str): Ruta del archivo HDF5 de salida.

    Returns:
        None
    """
    with h5py.File(checkpoint_path, "r") as f:
        data = f["model_weights"]

    with h5py.File(output_path, "w") as f:
        f.create_dataset("model_weights", data=data)

    print(f"Archivo de pesos convertido a HDF5: {output_path}")
    return output_path

def load_model() -> Sequential: 
    model = Sequential()
    model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer='orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))
    input_path = os.path.join("../models/checkpoint")
    output_path = os.path.join("../models/checkpoint.weights.h5")
    # print("FILE PATH:",file_path)
    file_path = convert_tf1_to_tf2(input_path,output_path)
    # model.load_weights(os.path.join('..','models','checkpoint'))
    checkpoint_status = tf.train.Checkpoint().restore(file_path)
    print("checkpoint_status",checkpoint_status)
    print("FILE_PATH",file_path)
    # checkpoint = tf.train.Checkpoint.restore(file_path)
    # print("CHECKPONIT",checkpoint) 
    # model.save_weights(checkpoint)
    # model.load_weights(checkpoint)
    # model2 = load_model2(os.path.join('..','models','checkpoint.model'))
    # model3 = TFSMLayer(os.path.join('..','models','checkpoint.model'),call_endpoint="seving_default")
    # Load the existing weights from previous checkpoint
    # model.save_weights(os.path.join('..','models','checkpoint.weights.h5'))
    # model.load_weights(os.path.join('..','models','checkpoint'))
    # model.load_weights(os.path.join('..','models','models - checkpoint 96.zip'))

    return model