from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dropout, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)
#from boto.dynamodb.batch import Batch

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ 
    Model 1
    Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    gru_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)

    # Add batch normalization 
    bn_rnn = BatchNormalization(name='rnn_bn')(gru_rnn)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ 
    Model 2
    Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # Add batch normalization
    bn_rnn = BatchNormalization(name='rnn_bn')(simp_rnn)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ 
    Model 3
    Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    # This code uses standard iteration to build the stacked network
    input_from_prev_layer = input_data
    for layer in range(0, recur_layers):
        # Add recurrent layers, each with batch normalization
        gru_rnn = GRU(units, return_sequences=True, implementation=2, name='rnn'+str(layer))(input_from_prev_layer)
        # Add batch normalization 
        bn_rnn = BatchNormalization(name='rnn_bn'+str(layer))(gru_rnn)
        input_from_prev_layer = bn_rnn
    
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(input_from_prev_layer)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ 
    Model 4
    Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, return_sequences=True, implementation=2, name='bidir_rnn'), 
                              merge_mode='concat')(input_data)

    # DON'T add batch normalization as it is not explicitly mentioned in the requirements 
    # bn_rnn = BatchNormalization(name='rnn_bn')(bidir_rnn)

    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)  
    
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, recur_layers=2, dilation=1, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    # Add dilational convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=1, 
                     padding=conv_border_mode,
                     dilation_rate=dilation,
                     activation='relu',
                     name='conv1d')(input_data)
    dropout = Dropout(0.3)(conv_1d)
    bn_cnn = BatchNormalization(name='bn_conv_1d')(dropout)

    # This code uses standard iteration to build the stacked network
    input_from_prev_layer = bn_cnn
    for layer in range(0, recur_layers):
        # Add recurrent layers, each with batch normalization
        gru_rnn = GRU(units, return_sequences=True, implementation=2, dropout=0.3, name='rnn'+str(layer))(input_from_prev_layer)
        # Add batch normalization 
        bn_rnn = BatchNormalization(name='rnn_bn'+str(layer))(gru_rnn)
        input_from_prev_layer = bn_rnn
    
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(input_from_prev_layer)
    
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    
    # Need to adjust the output length because of the convolutional layer
    model.output_length = lambda x: cnn_output_length(x, kernel_size, conv_border_mode, conv_stride, dilation=dilation)
    
    print(model.summary())
    return model

