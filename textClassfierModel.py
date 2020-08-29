#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing import text, sequence
from keras.utils import to_categorical
from keras.utils import multi_gpu_model
import keras
import os
import h5py
import re
import warnings
warnings.filterwarnings('ignore')

# model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),] )
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(rate)
        
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs)
        
        attn_output = self.dropout1(attn_output)
        
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        
        ffn_output = self.dropout2(ffn_output)
        
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    
    def __init__(self, maxlen, vocab_size, emded_dim):
        
        super(TokenAndPositionEmbedding, self).__init__()
        
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=emded_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=emded_dim)

    def call(self, x):
        
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        
        return x + positions

class NeuralNetwork(weight_name="transformer_model_weights.h5"):
    
    def __init__(self):
        pass
    
    
    def load_model(self):
        vocab_size = 40000  # Only consider the top 40k words
        embed_dim = 120  # Embedding size for each token
        num_heads = 8  # Number of attention heads
        ff_dim = 32  # Hidden layer size in feed forward network inside transformer
        max_len_padding=75

        inputs = layers.Input(shape=(max_len_padding,))
        dense_init = keras.initializers.he_uniform()

        embedding_layer = TokenAndPositionEmbedding(max_len_padding, vocab_size, embed_dim)

        x = embedding_layer(inputs)

        transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)

        x = transformer_block(x)

        x = keras.layers.Conv1D(128, 7, padding="valid")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)
        x = layers.Dropout(0.2)(x)

        x_1 = keras.layers.GlobalAveragePooling1D()(x)
        x_2 = keras.layers.GlobalMaxPooling1D()(x)
        x_a = keras.layers.Concatenate()([x_1,x_2])

        x=keras.layers.Dense(256, kernel_initializer=dense_init)(x_a)
        x=keras.layers.LeakyReLU()(x)
        x=keras.layers.Dropout(0.2)(x)

        x=keras.layers.Dense(128,kernel_initializer=dense_init)(x)
        x=keras.layers.LeakyReLU()(x)

        x_out=keras.layers.Dense(3827, activation="softmax",name="new_cat")(x)

        model=keras.models.Model(inputs, x_out)
        
        my_call_es = keras.callbacks.EarlyStopping(verbose=1, patience=5)
        my_call_cp = keras.callbacks.ModelCheckpoint('transformer_v3_singlegpu_{epoch:02d}.h5', save_weights_only=True, period=1)
        
        # my_opt = keras.optimizers.Nadam()
        model.compile(loss=['categorical_crossentropy'], optimizer="adam", metrics=['acc'])
        model.load_weights(weight_name)
        
        return model

