# File imports
import os

# Data Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Plotting Imports
import matplotlib.pyplot as plt
import random

# PCA Imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

# Encoder-Decoder Imports
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization
from keras.optimizers import Adam


class ModelHistory: ...

# +++++++++++++++++++++++++++++++++++++++++++++++++
# +++++            Encoder-Decoder            +++++
# +++++++++++++++++++++++++++++++++++++++++++++++++

class EncoderDecoderModel:
    def __init__(self, dataset: pd.DataFrame):
        """
        Encoder-Decoder Model constructor. Sets up model architecture, from here model can be trained or loaded.

        :dataset: Must be (rows, 108) shape DataFrame, with rows for ['gravity', 'temperature', 'c_o_ratio', 'metallicity']
        """
        self.dataset = dataset

        # Values will be encoded in a 2D space for clustering
        output_dim = 2

        # starting with 104 dimensions from spectra
        input_dim = 104

        # Encoder model 104 -> 52 -> 26 -> 10 -> 2 layers
        self.encoder = Sequential([
            Dense(52, activation='relu', input_shape=(input_dim,)),
            Dense(26, activation='relu'),
            Dense(10, activation='relu'),
            Dense(output_dim, activation='linear')
        ])

        # Decoder model 2 -> 20 -> 52 -> 104 layers
        self.decoder = Sequential([
            Dense(20, activation='relu', input_shape=(output_dim,)),
            Dense(52, activation='relu'),
            Dense(input_dim, activation='linear')
        ])

        # autoencoder - encodes & decodes the input, essentially 104 -> 2 -> 104 dimensions
        # Goal is to train this so decoder(encoder(X)) == X, and then use encoder for dim. reduction
        self.autoencoder = Model(inputs=self.encoder.input, outputs=self.decoder(self.encoder.output))

        # These will store our train & test sets if we decide to train
        self.X_train: pd.DataFrame
        self.X_test:  pd.DataFrame
        self.y_train: pd.DataFrame
        self.y_test:  pd.DataFrame

        self.X_train_scaled: np.ndarray
        self.X_test_scaled:  np.ndarray
        self.y_train_scaled: np.ndarray
        self.y_test_scaled:  np.ndarray

        # store our model's history
        self.model_history: ModelHistory


    def standardize_and_split_train_test(self, test_proportion: float = 0.2):
        """
        Split & standardize the dataset with a test ratio of the user's choice
        """
        # 80/20 by default train/test split
        train, test = train_test_split(self.dataset, test_size=0.2)

        # Input datasets
        self.X_train = train.drop(
            columns=['gravity', 
                    'temperature', 
                    'c_o_ratio', 
                    'metallicity'])

        self.X_test = test.drop(
            columns=['gravity', 
                    'temperature', 
                    'c_o_ratio', 
                    'metallicity'])

        # Output datasets
        self.y_train = train[['gravity', 'c_o_ratio', 'metallicity', 'temperature', ]]
        self.y_test  =  test[['gravity', 'c_o_ratio', 'metallicity', 'temperature', ]]

        # standardize X and Y values
        self.X_train_scaled = StandardScaler().fit_transform(self.X_train)
        self.X_test_scaled =  StandardScaler().fit_transform(self.X_test)
        self.y_train_scaled = StandardScaler().fit_transform(self.y_train)
        self.y_test_scaled =  StandardScaler().fit_transform(self.y_test)

    
    def train(self, epochs: int = 100, batch_size: int = 50) -> ModelHistory:
        """
        Train the model with the given dataset within the object.
        Data must be standardized & split into train & test before this
        Returns the autoencoder's model history
        """
        self.autoencoder.compile(loss='mse', optimizer=Adam())

        # model training
        model_history = self.autoencoder.fit(self.X_train_scaled, self.X_train_scaled,
                                             epochs=epochs, batch_size=batch_size, verbose=1,)
        self.model_history = model_history

        return model_history
    

    def load_encoder_weights(self, encoder_weights_path):
        """
        Load the encoder model from weights
        """
        return self.encoder.load_weights(encoder_weights_path)

    
    def load_decoder_weights(self, decoder_weights_path):
        """
        Load the decoder model from weights
        """
        return self.decoder.load_weights(decoder_weights_path)


    def save_encoder_weights(self, model_name):
        """
        Save encoder model weights
        """
        return self._save_model_weights(self.encoder, f'encoder_{model_name}')

    
    def save_decoder_weights(self, model_name):
        """
        Save decoder model weights
        """
        return self._save_model_weights(self.decoder, f'decoder_{model_name}')


    def plot_orig_vs_recon(self, title: str = ''):
        """
        With one random test dataset sample, plots the original & reconstructed values on the same graph
        """
        fig = plt.figure(figsize=(10,6))
        plt.suptitle(title)

        sample = random.choice(self.X_test_scaled)

        # Plot the reconstructed X values
        plt.plot(self.autoencoder.predict(sample.reshape(1, -1)).flatten(), label='reconstructed')

        # plot the original X values
        plt.plot(sample, label='original')

        plt.legend()
        plt.grid(True)

    
    def plot_encoded_against_single_feature(self, feature: str):
        """
        Plot the encoded scatterplot against the test dataset, with point darkness decreasing based on how relatively
        low that point's feature (temperature, gravity, etc.) is compared to the rest
        """
        encoded_X_test = self.encoder.predict(self.X_test_scaled, batch_size=self.X_test_scaled.shape[0])

        arr = self.y_test[feature].values
        normalized_values = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

        plt.scatter(encoded_X_test[:, 0], encoded_X_test[:, 1], s = 4, c=[(normalized_values[i], normalized_values[i], 0) for i in range(len(normalized_values))])
        plt.title(f'Autoencoder Graph - {feature}')
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')


    def _save_model_weights(self, model, model_name: str):
        """
        Save any model to reference data path
        """
        __reference_data_path__ = os.getenv("TelescopeML_reference_data")
        path_weights = os.path.join(__reference_data_path__,
                         f'trained_ML_models/trained_weights_{model_name}.h5',
                         )
        model.save_weights(path_weights)
        return path_weights




# +++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++              PCA              +++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++

class PCAModel:
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        PCA Model constructor. Calculates PCA values during construction
        
        :X: pandas dataframe with 104 columns, representing the spectrum values
        :y: pandas dataframe with columns ['gravity', 'c_o_ratio', 'metallicity', 'temperature']
        """
        self.X = X
        self.y = y
        self.X_scaled = StandardScaler().fit_transform(X)
        
        self.pca_df: pd.DataFrame
        self.variances: np.ndarray
        self.labels: list[str]
        
        self.pca_df, self.variances, self.labels = self._calculate_PCA_values()

    
    def graph_PC_variances(self, pc_count: int = 10):
        """
        Graph top X PCs that contribute most to the total variation in dataset. Shows as a bar chart. pc_count < num_rows is required.
        """
        variances = self.variances
        labels = self.labels

        # Since there are a lot of principal components let's look at the top 10
        # as the rest don't seem to have any effect at all
        plt.bar(x=range(1,pc_count+1), height=variances[:pc_count], tick_label=labels[:pc_count])
        plt.ylabel('Percentage of Explained Variance')
        plt.xlabel('Principal Component')
        plt.title(f'Scree Plot (Top {pc_count} PCs)')
        plt.show()


    def plot_PCA_against_single_feature(self, feature: str = ""):
        """
        Shows PCA scatterplot but darker dots represent lower values of this specific feature (temperature, gravity, etc.)
        """
        y = self.y
        pca_df = self.pca_df
        variances = self.variances

        # normalize Y values to be used for color shade
        arr = y[feature].values
        normalized_values = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

        # if no valid feature is given, then have all dots be the same color
        if feature not in ['temperature', 'c_o_ratio', 'gravity', 'metallicity']:
            normalized_values = np.array([0.5 for _ in range(len(normalized_values))])

        # draw scatterplot with top 2 PCs
        # using normalized values to determine how yellow dots are
        plt.scatter(pca_df.PC1, pca_df.PC2, s = 4, c=[(normalized_values[i], normalized_values[i], 0) for i in range(len(pca_df))])
        plt.title(f'PCA Graph - {feature}')
        plt.xlabel(f'PC1 - {variances[0]}%')
        plt.ylabel(f'PC2 - {variances[1]}%')


    def _calculate_PCA_values(self):
        """
        Calculates the PCA variances as a DF and stores it in pca_df. Only top 2 are really ever used
        """
        X_scaled = self.X_scaled
        y = self.y

        # Dimensionality reduction on X values
        pca = PCA()
        pca.fit(X_scaled)
        pca_data = pca.transform(X_scaled)

        # Calculate variances that each PC accounts for
        variances = np.round(pca.explained_variance_ratio_* 100, decimals=1)
        labels = [f'PC{i + 1}' for i in range(len(variances))]

        # convert PCA data to dataframe
        row_count = pca_data.shape[0]
        pca_df = pd.DataFrame(pca_data, index=list(range(row_count)), columns=labels)
        
        # return values to be used by the constructor
        return pca_df, variances, labels 

    