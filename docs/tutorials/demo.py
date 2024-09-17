import streamlit as st
import pandas as pd
import numpy as np
import os
from TelescopeML.UnsupervisedLearning import *
from TelescopeML.DataMaster import *


def create_data_processor():
    """
    Read CSV data from local reference path, and construct DataProcesor object from it
    """
    
    # read reference data from local CSV file
    __reference_data_path__ = os.getenv("TelescopeML_reference_data")
    train_BD = pd.read_csv(os.path.join(__reference_data_path__, 
                                        'training_datasets', 
                                        'browndwarf_R100_v4_newWL_v3.csv.bz2'), compression='bz2')

    # Training  variables
    X = train_BD.drop(
        columns=['gravity', 
                'temperature', 
                'c_o_ratio', 
                'metallicity'])


    # Target/Output feature variables
    y = train_BD[['gravity', 'c_o_ratio', 'metallicity', 'temperature', ]]
    y.loc[:, 'temperature'] = np.log10(y['temperature'])

    # relevant parameters for DataProcessor
    output_names = ['gravity', 'temperature', 'c_o_ratio', 'metallicity']
    wavelength_names = [item for item in train_BD.columns.to_list() if item not in output_names]
    wavelength_values = [float(item) for item in wavelength_names]

    # construct DataProcessor
    data_processor = DataProcessor( 
                                    flux_values=X.to_numpy(),
                                    wavelength_names=X.columns,
                                    wavelength_values=wavelength_values,
                                    output_values=y.to_numpy(),
                                    output_names=output_names,
                                    spectral_resolution=200,
                                    trained_ML_model=None,
                                    trained_ML_model_name='CNN',
                                    )
    return data_processor


def dp_scale_x_y(data_processor: DataProcessor):
    """
    Perform relevant operations to scale DataProcessor's X & Y data
    column-wise and row-wise as necessary
    """

    data_processor = create_data_processor()
    data_processor.split_train_validation_test(test_size=0.1, 
                                                val_size=0.1, 
                                                random_state_=42,)
    data_processor.standardize_X_row_wise()
    data_processor.standardize_y_column_wise()

    # train
    data_processor.X_train_min = data_processor.X_train.min(axis=1)
    data_processor.X_train_max = data_processor.X_train.max(axis=1)

    # validation
    data_processor.X_val_min = data_processor.X_val.min(axis=1)
    data_processor.X_val_max = data_processor.X_val.max(axis=1)

    # test
    data_processor.X_test_min = data_processor.X_test.min(axis=1)
    data_processor.X_test_max = data_processor.X_test.max(axis=1)

    df_MinMax_train = pd.DataFrame((data_processor.X_train_min, data_processor.X_train_max)).T
    df_MinMax_val = pd.DataFrame((data_processor.X_val_min, data_processor.X_val_max)).T
    df_MinMax_test = pd.DataFrame((data_processor.X_test_min, data_processor.X_test_max)).T

    df_MinMax_train.rename(columns={0:'min', 1:'max'}, inplace=True)
    df_MinMax_val.rename(columns={0:'min', 1:'max'}, inplace=True)
    df_MinMax_test.rename(columns={0:'min', 1:'max'}, inplace=True)

    data_processor.standardize_X_column_wise(
                                            output_indicator='Trained_StandardScaler_X_ColWise_MinMax',
                                            X_train = df_MinMax_train.to_numpy(),
                                            X_val   = df_MinMax_val.to_numpy(),
                                            X_test  = df_MinMax_test.to_numpy(),
                                            )
    return data_processor


def initialize_pca(data_processor):
    """
    Build & fit PCA model to X train
    """
    # Build PCA Model
    pca = PCAModel(
        X_train = data_processor.X_train_standardized_rowwise,
        X_val   = data_processor.X_val_standardized_rowwise,
        X_test  = data_processor.X_test_standardized_rowwise,
        y_train = data_processor.y_train_standardized_columnwise,
        y_val   = data_processor.y_val_standardized_columnwise,
        y_test  = data_processor.y_test_standardized_columnwise,
    )
    pca.build_model({'n_components': .999})
    return pca


def initialize_autoencoder(data_processor):
    """
    Build & fit autoencoder model to X train
    """
    # Load existing Autoencoder model
    autoencoder = Autoencoder(
        X_train = data_processor.X_train_standardized_rowwise,
        X_val   = data_processor.X_val_standardized_rowwise,
        X_test  = data_processor.X_test_standardized_rowwise,
        y_train = data_processor.y_train_standardized_columnwise,
        y_val   = data_processor.y_val_standardized_columnwise,
        y_test  = data_processor.y_test_standardized_columnwise,
    )
    config = {
        'output_dim': 20,
        'encoder_layers': [60, 35],
        'decoder_layers': [35, 60],
        # Exclude final input & output layer for each
    }
    autoencoder.build_model(config)
    return autoencoder


"""
# Unsupervised Learning Visualized

Abhinav Uppala
"""

# Initialize DataProcessor object
data_processor = create_data_processor()
data_processor = dp_scale_x_y(data_processor)


# Dropdown
vis_dropdown = st.selectbox(label='Pick visualization',
                            options=['PCA', 'Autoencoder', 'GMM'],
                            index=None,
                            key='visualization')

if st.session_state.get('visualization'):
    
    if vis_dropdown == 'PCA':
        """
        ## PCA for Dimensionality Reduction

        PCA (Principal Component Analysis) is a method of reducing the input's dimensions, while also preserving important features and variation in the data.
        This specific model aims to preserve at least 99.9% of the variation in the training data. This demo shows its effectiveness by comparing a
        random test X value, along with the same PCA-compressed & reconstructed X value, demonstrating how features are preserved.
        """
        pca = initialize_pca(data_processor)
        index = np.random.randint(data_processor.X_test_standardized_rowwise.shape[0])

        actual = data_processor.X_test_standardized_rowwise[index].reshape(1, -1)

        encoded = pca.encode_input_data(actual)
        estimated = pca.decode_output_data(encoded)

        st.line_chart(pd.DataFrame(
            {'Actual': actual[0], 'Reconstructed': estimated[0]}
        ))

    
    elif vis_dropdown == 'Autoencoder':
        """
        ### Autoencoder for Dimensionality Reduction

        Autoencoders are models with 2 parts - encoder & decoder, used to transform an input into a different state and transform it back. Training the encoder and
        decoder ensures the model is able to identify trends and variability in the original data and preseve them in a different format. By encoding our input data into
        a lower dimension (104 -> 20 in this case), we can perform dimensionality reduction while retaining the original features of the input. This demo showcases this
        by showing a randomly selected X value along with it's reconstruction from encoding & decoding.
        """
        autoencoder = initialize_autoencoder(data_processor)
        index = np.random.randint(data_processor.X_test_standardized_rowwise.shape[0])

        actual = data_processor.X_test_standardized_rowwise[index].reshape(1, -1)

        # Compare actual & reconstructed before training
        encoded_no_train = autoencoder.encoder.predict(actual)
        estimated_no_train = autoencoder.decoder.predict(encoded_no_train)

        # Load saved weights
        autoencoder.load_from_indicator('20d_correct_scaling_100epochs')

        # Compare actual & reconstructed after training
        encoded = autoencoder.encoder.predict(actual)
        estimated = autoencoder.decoder.predict(encoded)

        st.line_chart(pd.DataFrame(
            {'Actual': actual[0],
             'Reconstructed (Before Training)': estimated_no_train[0],
             'Reconstructed (After Training)': estimated[0]}
        ))


    else:
        """
        ### TODO

        Not yet implemented!
        """