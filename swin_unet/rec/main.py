# Fixed code with debug logging for ID mapping
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import logging
import traceback
from typing import List, Dict, Any, Optional
import json
import requests
import random
import wandb
from tensorflow.keras.callbacks import Callback
import time
from datetime import datetime
import pandas as pd
import merlin.models.tf as mm
from merlin.schema import Schema, Tags, ColumnSchema
from merlin.io import Dataset
from merlin.models.utils.dataset import unique_rows_by_features
from tqdm import tqdm
import pg8000
import boto3
from sklearn.model_selection import train_test_split
import shutil
import gc
import glob
from merlin.models.tf import TwoTowerModelV2, Encoder, MLPBlock, ContrastiveOutput
from merlin.models.tf.outputs.base import DotProduct
import merlin.models.tf as mm
import tensorflow as tf
from merlin.schema import Tags

import logging

# Change INFO to DEBUG to show debug messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# S3 configuration
S3_BUCKET = "smokeshow-recommender-dataset"
S3_PREFIX = "embeddings/"

# Initialize S3 client and embedding cache
s3 = boto3.client('s3')
embedding_cache = {}

class WandbCallback(Callback):
    def __init__(self, validation_data=None, batch_size=None):
        super().__init__()
        self.validation_data = validation_data
        self.batch_size = batch_size
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_metrics = {
            "train_loss": logs.get("loss"),
            "train_ndcg_at_5": logs.get("ndcg_at_5"),
            "train_recall_at_5": logs.get("recall_at_5"),
            "train_regularization_loss": logs.get("regularization_loss")
        }
        if "recall_at_10" in logs:
            train_metrics["train_recall_at_10"] = logs.get("recall_at_10")
        if "ndcg_at_10" in logs:
            train_metrics["train_ndcg_at_10"] = logs.get("ndcg_at_10")
        
        train_metrics = {k: v for k, v in train_metrics.items() if v is not None}
        wandb.log(train_metrics, step=epoch + 1)
        
        if self.validation_data and self.batch_size:
            val_metrics = self.model.evaluate(
                self.validation_data,
                batch_size=self.batch_size,
                return_dict=True
            )
            filtered_val_metrics = {
                "val_loss": val_metrics.get("loss"),
                "val_ndcg_at_5": val_metrics.get("ndcg_at_5"),
                "val_recall_at_5": val_metrics.get("recall_at_5"),
                "val_regularization_loss": val_metrics.get("regularization_loss")
            }
            if "recall_at_10" in val_metrics:
                filtered_val_metrics["val_recall_at_10"] = val_metrics.get("recall_at_10")
            if "ndcg_at_10" in val_metrics:
                filtered_val_metrics["val_ndcg_at_10"] = val_metrics.get("ndcg_at_10")
            
            filtered_val_metrics = {k: v for k, v in filtered_val_metrics.items() if v is not None}
            wandb.log(filtered_val_metrics, step=epoch + 1)

@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class TFUnifiedUserEmbeddingModel(tf.keras.Model):
    def __init__(
        self,
        video_embedding_dim=256,
        user_feature_dim=32,
        video_meta_dim=1538,
        interaction_feature_dim=12,
        hidden_dim=512,
        max_sequence_length=50,
        num_heads=8,
        num_layers=2,
        dropout=0.0,
        l2_reg=0.05,
        *args, **kwargs
    ):   
        super().__init__(*args, **kwargs)
        self.video_embedding_dim = video_embedding_dim
        self.user_feature_dim = user_feature_dim
        self.video_meta_dim = video_meta_dim
        self.interaction_feature_dim = interaction_feature_dim
        self.hidden_dim = hidden_dim
        self.max_sequence_length = max_sequence_length
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.l2_reg = l2_reg
        
        self.interaction_feature_encoder = tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.video_meta_encoder = tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.embedding_layer = tf.keras.layers.Dense(hidden_dim, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name="embedding_layer")
        
        total_input_dim = video_embedding_dim + 256 + 256
        self.input_projector = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, input_shape=(total_input_dim,), kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(dropout)
        ])
        
        self.position_encoding_var = self.add_weight(
            "position_encoding",
            shape=[1, max_sequence_length, hidden_dim],
            initializer=tf.keras.initializers.Zeros(),
            trainable=True
        )
        self._initialize_position_encoding()
        
        self.transformer_layers = [
            tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_dim // num_heads, dropout=dropout)
            for _ in range(num_layers)
        ]
        self.ffns = [
            tf.keras.Sequential([
                tf.keras.layers.Dense(hidden_dim * 4, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(hidden_dim, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
            ]) for _ in range(num_layers)
        ]
        self.attention_layer_norms = [tf.keras.layers.LayerNormalization() for _ in range(num_layers)]
        self.ffn_layer_norms = [tf.keras.layers.LayerNormalization() for _ in range(num_layers)]
        
        self.attention_pooling = tf.keras.layers.Dense(1)
        self.output_projector = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(video_embedding_dim)
        ])
    
    def _initialize_position_encoding(self):
        max_len = self.max_sequence_length
        d_model = self.position_encoding_var.shape[-1]
        position = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * (-tf.math.log(10000.0) / d_model))
        sin_enc = tf.sin(position * div_term)
        cos_enc = tf.cos(position * div_term)
        pos_enc = tf.stack([sin_enc, cos_enc], axis=2)
        pos_enc = tf.reshape(pos_enc, [1, max_len, d_model])
        self.position_encoding_var.assign(pos_enc)
    
    def call(self, inputs, training=False):
        # Extract flattened inputs
        video_embeddings_flat = inputs['video_embeddings']          
        interaction_features_flat = inputs['interaction_features']  
        video_meta_features_flat = inputs['video_meta_features']    
        sequence_lengths = inputs.get('sequence_lengths')
        
        # Reshape flattened inputs back to 2D
        batch_size = tf.shape(video_embeddings_flat)[0]
        video_embeddings = tf.reshape(video_embeddings_flat, 
                                    [batch_size, self.max_sequence_length, 256])
        interaction_features = tf.reshape(interaction_features_flat, 
                                        [batch_size, self.max_sequence_length, 12])
        video_meta_features = tf.reshape(video_meta_features_flat, 
                                        [batch_size, self.max_sequence_length, 1538])
        
        # Compute sequence lengths if not provided
        if sequence_lengths is None:
            sequence_lengths = tf.reduce_sum(
                tf.cast(tf.reduce_any(video_embeddings != 0, axis=-1), tf.int32), axis=1
            )
        
        # Encode interaction and meta features
        encoded_interaction = self.interaction_feature_encoder(interaction_features)
        encoded_meta = self.video_meta_encoder(video_meta_features)
        
        # Combine features
        combined_features = tf.concat([video_embeddings, encoded_interaction, encoded_meta], axis=-1)
        
        # Flatten for input projector
        seq_len = tf.shape(combined_features)[1]
        combined_features_flat = tf.reshape(combined_features, [-1, self.video_embedding_dim + 256 + 256])
        
        # Apply input projector
        x = self.input_projector(combined_features_flat, training=training)
        x = tf.reshape(x, [batch_size, seq_len, -1])
        
        # Add positional encoding
        positions = self.position_encoding_var[:, :seq_len, :]
        x_embedded = self.embedding_layer(x)
        x = x_embedded + positions
        
        # Create padding mask
        padding_mask = tf.sequence_mask(sequence_lengths, maxlen=seq_len, dtype=tf.float32)
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]
        
        # Apply transformer layers
        for i in range(self.num_layers):
            attn_output = self.transformer_layers[i](
                x, x, attention_mask=padding_mask, training=training
            )
            x = x + attn_output
            x = self.attention_layer_norms[i](x)
            ffn_output = self.ffns[i](x, training=training)
            x = x + ffn_output
            x = self.ffn_layer_norms[i](x)
        
        # Attention pooling
        attention_scores = self.attention_pooling(x)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        user_embedding = tf.reduce_sum(x * attention_weights, axis=1)
        
        # Apply output projector
        user_embedding = self.output_projector(user_embedding, training=training)
        return user_embedding

class UserEmbeddingGenerator:
    def __init__(
        self,
        video_embedding_dim=256,
        user_embedding_dim=256,
        video_meta_dim=1538,
        interaction_feature_dim=12,
        hidden_dim=512,
        max_sequence_length=50,
        video_embeddings_path=None,
        reverse_mapping_path=None,
        model_path=None,
        faiss_api_url=None,
        api_key=None,
        dropout=0.1
    ):
        self.video_embedding_dim = video_embedding_dim
        self.user_embedding_dim = user_embedding_dim
        self.video_meta_dim = video_meta_dim
        self.interaction_feature_dim = interaction_feature_dim
        
        self.video_embeddings = None
        self.reverse_mapping = None
        if video_embeddings_path and reverse_mapping_path:
            self.load_video_embeddings(video_embeddings_path, reverse_mapping_path)
        
        self.model = TFUnifiedUserEmbeddingModel(
            video_embedding_dim=video_embedding_dim,
            user_feature_dim=32,
            video_meta_dim=video_meta_dim,
            interaction_feature_dim=interaction_feature_dim,
            hidden_dim=hidden_dim,
            max_sequence_length=max_sequence_length,
            dropout=dropout
        )
        
        if model_path:
            self.load_model(model_path)
        
        self.user_embedding_cache = {}
        self.faiss_api_url = faiss_api_url
        self.api_key = api_key
    
    def load_video_embeddings(self, video_embeddings_path, reverse_mapping_path):
        self.video_embeddings = np.load(video_embeddings_path)
        self.reverse_mapping = np.load(reverse_mapping_path)
    
    def _get_video_embedding(self, video_id):
        if self.video_embeddings is not None and video_id in self.reverse_mapping:
            idx = self.reverse_mapping[video_id]
            return self.video_embeddings[idx]
        return np.zeros(self.video_embedding_dim, dtype=np.float32)
    
    def _extract_interaction_features(self, interaction):
        return np.zeros(self.interaction_feature_dim, dtype=np.float32)
    
    def _create_contextual_embedding(self, user_id, current_time=None):
        if current_time is None:
            current_time = datetime.now()
        
        hour = current_time.hour
        day_of_week = current_time.weekday()
        
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin = np.sin(2 * np.pi * day_of_week / 7)
        dow_cos = np.cos(2 * np.pi * day_of_week / 7)
        
        context_vector = np.array([hour_sin, hour_cos, dow_sin, dow_cos] + [0] * (self.interaction_feature_dim - 4), 
                                 dtype=np.float32)
        
        video_embedding = np.zeros(self.video_embedding_dim, dtype=np.float32)
        video_meta_vector = np.zeros(self.video_meta_dim, dtype=np.float32)
        
        inputs = {
            'video_embeddings': tf.expand_dims(tf.convert_to_tensor([video_embedding]), axis=0),
            'interaction_features': tf.expand_dims(tf.convert_to_tensor([context_vector]), axis=0),
            'video_meta_features': tf.expand_dims(tf.convert_to_tensor([video_meta_vector]), axis=0)
        }
        
        embedding = self.model(inputs)
        return embedding.numpy().flatten()
    
    def compute_user_embedding(self, user_id, interactions, current_time=None):
        cache_key = f"{user_id}_{len(interactions)}"
        if cache_key in self.user_embedding_cache:
            return self.user_embedding_cache[cache_key]
        
        if not interactions:
            logger.info(f"No interactions for user {user_id}, generating contextual embedding")
            embedding = self._create_contextual_embedding(user_id, current_time)
            self.user_embedding_cache[cache_key] = embedding
            return embedding
        
        if 'exposed_time' in interactions[0]:
            sorted_interactions = sorted(interactions, key=lambda x: x.get('exposed_time', 0))
        else:
            sorted_interactions = interactions
        
        recent_interactions = sorted_interactions[-self.model.max_sequence_length:]
        processed_interactions = []
        
        for interaction in recent_interactions:
            video_id = interaction.get('video_id', interaction.get('pid', None))
            if video_id is None:
                continue
            
            if 'video_embedding' not in interaction:
                interaction['video_embedding'] = self._get_video_embedding(video_id)
            
            if 'interaction_vector' not in interaction:
                interaction['interaction_vector'] = self._extract_interaction_features(interaction)
            
            if 'video_meta_vector' not in interaction and 'video_meta' in interaction:
                interaction['video_meta_vector'] = create_video_meta_vector(interaction['video_meta'])
            
            processed_interactions.append(interaction)
        
        if not processed_interactions:
            logger.info(f"No valid interactions for user {user_id}, generating contextual embedding")
            embedding = self._create_contextual_embedding(user_id, current_time)
            self.user_embedding_cache[cache_key] = embedding
            return embedding
        
        video_embeddings = np.array([inter['video_embedding'] for inter in processed_interactions])
        interaction_features = np.array([inter['interaction_vector'] for inter in processed_interactions])
        video_meta_features = np.array([inter['video_meta_vector'] for inter in processed_interactions])
        
        video_embeddings_tf = tf.convert_to_tensor(video_embeddings, dtype=tf.float32)
        interaction_features_tf = tf.convert_to_tensor(interaction_features, dtype=tf.float32)
        video_meta_features_tf = tf.convert_to_tensor(video_meta_features, dtype=tf.float32)
        
        video_embeddings_tf = tf.expand_dims(video_embeddings_tf, axis=0)
        interaction_features_tf = tf.expand_dims(interaction_features_tf, axis=0)
        video_meta_features_tf = tf.expand_dims(video_meta_features_tf, axis=0)
        
        inputs = {
            'video_embeddings': video_embeddings_tf,
            'interaction_features': interaction_features_tf,
            'video_meta_features': video_meta_features_tf
        }
        
        user_embedding = self.model(inputs)
        user_embedding = user_embedding.numpy().flatten()
        
        self.user_embedding_cache[cache_key] = user_embedding
        
        logger.info(f"Sample from user embedding: {user_embedding[:5]}")
        
        return user_embedding

class MerlinRecommender:
    def __init__(
        self,
        embedding_dim=256,
        model_dir="models/merlin_twotower",
        user_embedding_generator=None,
        video_feature_manager=None,
        dropout=0.05,
        l2_reg=0.05
    ):
        self.embedding_dim = embedding_dim
        self.model_dir = model_dir
        self.user_embedding_generator = user_embedding_generator
        self.video_feature_manager = video_feature_manager
        self.model = None
        self.schema = None
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.user_embedding_model = TFUnifiedUserEmbeddingModel(
            video_embedding_dim=embedding_dim,
            user_feature_dim=32,
            video_meta_dim=1538,
            interaction_feature_dim=12,
            hidden_dim=512,
            max_sequence_length=50,
            dropout=dropout,
            l2_reg=l2_reg
        )
        self.user_embedding_cache = {}
        self.video_embedding_cache = {}
        
        os.makedirs(model_dir, exist_ok=True)
        
        self.device = 'cuda' if tf.test.is_gpu_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        self.video_projection = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', input_shape=(1794,)),
            tf.keras.layers.Dense(256),
            tf.keras.layers.LayerNormalization()
        ])
    
    def _create_schema(self):
        user_cardinality = 100
        video_cardinality = 159
        max_sequence_length = self.user_embedding_model.max_sequence_length
        embedding_dim = self.embedding_dim
        
        self.schema = Schema([
            ColumnSchema(
                name="sequence_lengths",
                dtype=np.int32,
                tags=[Tags.USER]
            ),
            ColumnSchema(
                name="video_embeddings",
                dtype=np.float32,
                dims=(max_sequence_length, 256),
                tags=[Tags.USER, Tags.CONTINUOUS, Tags.SEQUENCE]
            ),
            ColumnSchema(
                name="interaction_features",
                dtype=np.float32,
                dims=(max_sequence_length, 12),
                tags=[Tags.USER, Tags.CONTINUOUS, Tags.SEQUENCE]
            ),
            ColumnSchema(
                name="video_meta_features",
                dtype=np.float32,
                dims=(max_sequence_length, 1538),
                tags=[Tags.USER, Tags.CONTINUOUS, Tags.SEQUENCE]
            ),
            ColumnSchema(
                name="video_embedding",
                dtype=np.float32,
                dims=(embedding_dim,),
                tags=[Tags.ITEM, Tags.CONTINUOUS]
            ),
            ColumnSchema(
                name="user_id",
                dtype=np.int64,
                properties={"domain": {"min": 1, "max": user_cardinality}},
                tags=[Tags.USER_ID, Tags.CATEGORICAL]
            ),
            ColumnSchema(
                name="video_id",
                dtype=np.int64,
                properties={"domain": {"min": 1, "max": video_cardinality}},
                tags=[Tags.ITEM_ID, Tags.CATEGORICAL]
            ),
        ])
        return self.schema
    
    def _build_model(self):
        if self.schema is None:
            self._create_schema()
        
        user_schema = self.schema.select_by_tag(Tags.USER)
        user_input = mm.InputBlockV2(user_schema, aggregation=None)
        user_encoder = Encoder(user_input, self.user_embedding_model)
        
        item_schema = self.schema.select_by_tag(Tags.ITEM)
        item_input = mm.InputBlockV2(item_schema, aggregation='concat')
        item_tower = MLPBlock(
            [512, 256],
            activation='gelu',
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
            dropout=self.dropout
        )
        item_encoder = Encoder(item_input, item_tower)
                
        logger.info("Creating TwoTowerModel...")
        
        output = ContrastiveOutput(
            schema=self.schema.select_by_tag(Tags.ITEM_ID),  
            to_call=DotProduct(),
            negative_samplers='in-batch'
        )
        
        self.model = TwoTowerModelV2(
            query_tower=user_encoder,
            candidate_tower=item_encoder,
            outputs=[output],
            schema=self.schema
        )
        
        retrieval_metrics = [mm.RecallAt(k) for k in [10]] + [mm.NDCGAt(k) for k in [10]]
        
        # Use LazyAdam with a fixed learning rate 
        # This is the simplest approach without trying to use learning rate schedules
        try:
            optimizer = mm.LazyAdam(learning_rate=0.0001)
            logger.info("Using LazyAdam optimizer")
        except Exception as e:
            logger.warning(f"LazyAdam not available, falling back to standard Adam: {str(e)}")
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        self.model.compile(
            optimizer=optimizer,
            run_eagerly=False,
            metrics=retrieval_metrics,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        )
        
        logger.info("TensorFlow TwoTowerModel built successfully")
        return self.model
    def save_model(self):
        """
        Save model components needed for recommendation deployment.
        
        This function saves the full model and also extracts and saves:
        1. Query tower (user encoder)
        2. User features
        3. Item features
        4. User embeddings
        5. Item embeddings
        
        These components are needed for efficient recommendation generation.
        """
        if not self.model:
            logger.warning("No model to save!")
            return False
        
        try:
            # Create model directories
            # If directory exists, makedirs will not raise error (exist_ok=True)
            # If directory exists, makedirs will overwrite it (exist_ok=True)
            os.makedirs(self.model_dir, exist_ok=True)  # Creates new dir or uses existing
            data_folder = os.path.join(self.model_dir, "data")
            os.makedirs(data_folder, exist_ok=True)
            
            # 1. Save the main model
            self.model.save(self.model_dir)
            logger.info(f"Model saved to {self.model_dir}")
            
            # 2. Extract and save the query tower (user encoder)
            query_tower = self.model.query_encoder
            query_tower_path = os.path.join(data_folder, "query_tower")
            query_tower.save(query_tower_path)
            logger.info(f"Query tower saved to {query_tower_path}")
            
            # 3. Extract and save item and user features
            train_dataset = self._get_latest_training_dataset()
            if train_dataset is not None:
                # Extract unique user features
                logger.info("Extracting unique user features")
                user_features = unique_rows_by_features(train_dataset, Tags.USER, Tags.USER_ID).compute().reset_index(drop=True)
                user_features_path = os.path.join(data_folder, "user_features.parquet")
                user_features.to_parquet(user_features_path)
                logger.info(f"User features saved to {user_features_path} ({len(user_features)} users)")
                
                # Extract unique item features
                logger.info("Extracting unique item features")
                item_features = unique_rows_by_features(train_dataset, Tags.ITEM, Tags.ITEM_ID).compute().reset_index(drop=True)
                item_features_path = os.path.join(data_folder, "item_features.parquet")
                item_features.to_parquet(item_features_path)
                logger.info(f"Item features saved to {item_features_path} ({len(item_features)} items)")
                
                # 4. Generate and save item embeddings
                logger.info("Generating item embeddings")
                item_embeddings = self.model.candidate_embeddings(
                    Dataset(item_features, schema=self.schema.select_by_tag(Tags.ITEM)),
                    batch_size=256,
                    index=Tags.ITEM_ID
                )
                item_embs_df = item_embeddings.compute(scheduler="synchronous")
                item_embs_path = os.path.join(data_folder, "item_embeddings.parquet")
                item_embs_df.to_parquet(item_embs_path)
                logger.info(f"Item embeddings saved to {item_embs_path}")
                
                # 5. Generate and save user embeddings
                logger.info("Generating user embeddings")
                user_embeddings = self.model.query_embeddings(
                    Dataset(user_features, schema=self.schema.select_by_tag(Tags.USER)),
                    batch_size=256,
                    index=Tags.USER_ID
                )
                user_embs_df = user_embeddings.compute(scheduler="synchronous").reset_index()
                user_embs_path = os.path.join(data_folder, "user_embeddings.parquet")
                user_embs_df.to_parquet(user_embs_path)
                logger.info(f"User embeddings saved to {user_embs_path}")
            else:
                logger.warning("No training dataset available, skipping feature and embedding extraction")
            
            # Save additional metadata needed for recommendations
            metadata = {
                "embedding_dim": self.embedding_dim,
                "max_sequence_length": self.user_embedding_model.max_sequence_length,
                "model_type": "TwoTowerModelV2",
                "saved_date": datetime.now().isoformat(),
            }
            
            # Save ID mappings if available
            if hasattr(self, 'user_id_map') and hasattr(self, 'video_id_map'):
                # Save ID mappings separately as they can be large
                with open(os.path.join(data_folder, 'user_id_map.json'), 'w') as f:
                    json.dump(self.user_id_map, f)
                with open(os.path.join(data_folder, 'video_id_map.json'), 'w') as f:
                    json.dump(self.video_id_map, f)
                logger.info(f"Saved ID mappings ({len(self.user_id_map)} users, {len(self.video_id_map)} videos)")
                
                # Add mapping sizes to metadata
                metadata["user_id_map_size"] = len(self.user_id_map)
                metadata["video_id_map_size"] = len(self.video_id_map)
            
            # Save metadata
            with open(os.path.join(data_folder, 'model_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model metadata saved to {data_folder}/model_metadata.json")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model components: {str(e)}")
            logger.error(traceback.format_exc())
            return False
        
    def _get_latest_training_dataset(self):
        """
        Get the latest training dataset.
        
        Returns:
            The training dataset or None if not available
        """
        try:
            # Check for processed training data
            train_dir = "processed_data_train"  # Adjust as needed
            if os.path.exists(train_dir):
                parquet_files = glob.glob(os.path.join(train_dir, "*.parquet"))
                if parquet_files:
                    logger.info(f"Found {len(parquet_files)} parquet files in {train_dir}")
                    return Dataset(parquet_files, schema=self.schema)
            
            logger.warning(f"No training data found in {train_dir}")
            return None
        except Exception as e:
            logger.error(f"Error getting training dataset: {str(e)}")
            return None
    
    
    def get_video_embedding(self, video_id):
        if video_id in self.video_embedding_cache:
            return self.video_embedding_cache[video_id]
        
        if self.video_feature_manager and hasattr(self.video_feature_manager, 'get_video_embedding'):
            try:
                video_embedding = self.video_feature_manager.get_video_embedding(video_id)
                self.video_embedding_cache[video_id] = video_embedding
                return video_embedding
            except Exception as e:
                logger.warning(f"Failed to get video embedding for {video_id}: {str(e)}")
        
        return np.random.randn(self.embedding_dim).astype(np.float32)

    def clear_caches(self):
        """Clear all caches to free up memory."""
        global embedding_cache
        embedding_cache.clear()
        self.user_embedding_cache.clear()
        self.video_embedding_cache.clear()
        if hasattr(self.user_embedding_generator, 'user_embedding_cache'):
            self.user_embedding_generator.user_embedding_cache.clear()
        logger.info("All caches cleared")
    
    def process_interactions(self, interaction_dict, split_name, batch_size=64, output_dir=None, 
                            user_id_map=None, video_id_map=None):
        """
        Process interactions with memory optimization.
        """
        # Setup output directory
        output_dir = self._setup_output_directory(output_dir or f"processed_data_{split_name}")
        
        # Initialize processing variables
        logger.info(f"Preparing {split_name} data for {len(interaction_dict)} users")
        batch_idx = 0

        
        # Process in chunks to avoid memory issues
        chunk_size = 500 # Process 1000 users at a time
        all_users = list(interaction_dict.keys())
        
        # Validate and create ID mappings if needed
        video_ids = self._collect_video_ids(interaction_dict)
        user_id_map, video_id_map = self._ensure_id_mappings(interaction_dict, video_ids, user_id_map, video_id_map, split_name)
        
        for chunk_start in range(0, len(all_users), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(all_users))
            chunk_users = all_users[chunk_start:chunk_end]
            
            processed_data = []
            
            for user_id in tqdm(chunk_users, desc=f"Processing {split_name} chunk {chunk_start//chunk_size + 1}"):
                interactions = interaction_dict[user_id]
                
                # Ensure interactions have valid video_meta_vector
                self._validate_interaction_vectors(user_id, interactions)
                
                # Sort interactions by timestamp
                sorted_interactions = sorted(interactions, key=lambda x: x.get('timestamp', 0))
                
                # DEBUG - show interaction count before filtering
                # logger.info(f"User {user_id}: {len(sorted_interactions)} sorted interactions")
                
                # Now correctly create history/target split ensuring history isn't empty
                if len(sorted_interactions) > 5:
                    # Calculate target ratio (unchanged)
                    if int(len(sorted_interactions)) > 100:
                        target_ratio = 0.15  # 20% targets
                    else:
                        target_ratio = 0.1  # 10% targets
                        
                    # FIXED: Calculate target count but ALWAYS ensure at least 2 items remain for history
                    min_history = 2  # Minimum required history items
                    num_target = min(
                        int(len(sorted_interactions) * target_ratio),  # Normal calculation
                        len(sorted_interactions) - min_history  # But ensure min_history left for history
                    )
                    num_target = max(1, num_target)  # At least 1 target
                    
                    # Get history and target - history is now guaranteed to have at least min_history items
                    history = sorted_interactions[:-num_target]
                    target = sorted_interactions[-num_target:]
                    
                    # logger.info(f"User {user_id}: Split into {len(history)} history and {len(target)} target interactions")
                    
                    # Double-check that we have history items (should always pass now)
                    if not history:
                        logger.error(f"ERROR: History should not be empty! User {user_id}: "
                                    f"{len(sorted_interactions)} interactions, num_target={num_target}")
                        continue
                    
                    batch_data = self._process_single_interaction(user_id, history, target, user_id_map, video_id_map)
                    if batch_data:
                        processed_data.append(batch_data)
                else:
                    logger.info(f"Skipping user {user_id} due to insufficient interactions: {len(sorted_interactions)}")
            if processed_data:
                self._save_batch(processed_data, output_dir, batch_idx, split_name)
                batch_idx += 1
            
            # Clear memory
            processed_data.clear()
            gc.collect()
        
        # Log final statistics
        logger.info(f"Processed all chunks for {split_name} split")
        
        # Validate the processed files
        self._validate_processed_files(output_dir, user_id_map, video_id_map, split_name)
        
        return output_dir, user_id_map, video_id_map
    
    def _process_single_interaction(self, user_id, history, target, user_id_map, video_id_map):
        """
        Process a single interaction with history for validation/test.
        """
        # DEBUG LOGGING
        logger.debug(f"_process_single_interaction for user {user_id}: history length={len(history)}, "
                    f"target type={type(target).__name__}")
        
        # Check if history needs flattening
        flat_history = []
        for item in history:
            if isinstance(item, list):
                flat_history.extend(item)
            else:
                flat_history.append(item)
        
        # DEBUG - check if history has expected fields
        if flat_history:
            sample_keys = list(flat_history[0].keys()) if flat_history else []
            logger.info(f"User {user_id}: flat_history length={len(flat_history)}, sample keys={sample_keys}")
        else:
            logger.warning(f"Empty history for user {user_id}")
        
        # Ensure target is a dictionary
        if isinstance(target, list):
            if target:  # Non-empty list
                target = target[0]  # Take first element
                logger.debug(f"Target is a list, using first element")
            else:
                logger.warning(f"Empty target list for user {user_id}")
                return None
        
        # *** IMPORTANT: Check that we have both history and target ***
        if not flat_history:
            logger.warning(f"User {user_id} has no history interactions, cannot create valid training example")
            return None
        
        # Ensure embeddings
        self._ensure_embeddings(user_id, flat_history + [target])
        
        # Create padded features
        flat_video_emb, flat_interact, flat_meta, seq_len = self._create_padded_features(flat_history)
        
        # Map IDs
        mapped_user_id, mapped_video_id = self._map_ids(user_id, target['video_id'], user_id_map, video_id_map, 'eval')
        
        if mapped_user_id is not None and mapped_video_id is not None:
            return {
                'user_id': mapped_user_id,
                'video_id': mapped_video_id,
                'video_embeddings': flat_video_emb,
                'interaction_features': flat_interact,
                'video_meta_features': flat_meta,
                'video_embedding': target['video_embedding'],
                'sequence_lengths': seq_len
            }
        return None
    
    def _setup_output_directory(self, output_dir):
        """Setup and clean the output directory."""
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            logger.info(f"Deleted existing output directory: {output_dir}")
        else:
            logger.info("No existing output directory found, creating new")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def _collect_video_ids(self, interaction_dict):
        """Collect all video IDs from interactions."""
        video_ids = set()
        for interactions in interaction_dict.values():
            for inter in interactions:
                video_id = str(inter.get('video_id'))
                if video_id:
                    video_ids.add(video_id)
                else:
                    logger.warning(f"Empty or missing video_id in interaction for user {inter.get('user_id')}: {inter}")
        logger.debug(f"Collected {len(video_ids)} unique video IDs")
        return video_ids
    
    def _ensure_id_mappings(self, interaction_dict, video_ids, user_id_map, video_id_map, split_name):
        """Ensure valid ID mappings are available, creating them if necessary."""
        if user_id_map is None:
            user_ids = sorted(set(interaction_dict.keys()))
            user_id_map = {user_id: idx + 1 for idx, user_id in enumerate(user_ids)}
            logger.debug(f"Created user_id_map with {len(user_id_map)} users")
        
        if video_id_map is None:
            video_id_map = {video_id: idx + 1 for idx, video_id in enumerate(sorted(video_ids))}
            logger.debug(f"Created video_id_map with {len(video_id_map)} videos")
        
        logger.info(f"Using ID mappings for {split_name}: {len(user_id_map)} users, {len(video_id_map)} videos")
        
        # Debug: Print first 10 entries of each mapping
        logger.debug(f"Sample user_id_map (first 10): {dict(list(user_id_map.items())[:10])}")
        logger.debug(f"Sample video_id_map (first 10): {dict(list(video_id_map.items())[:10])}")
        
        # Check for unmapped IDs
        unmapped_users = [uid for uid in interaction_dict if str(uid) not in user_id_map]
        unmapped_videos = [vid for vid in video_ids if str(vid) not in video_id_map]
        
        if unmapped_users:
            logger.error(f"Found {len(unmapped_users)} unmapped user IDs: {unmapped_users[:10]}...")
        if unmapped_videos:
            logger.error(f"Found {len(unmapped_videos)} unmapped video IDs: {unmapped_videos[:10]}...")
        
        return user_id_map, video_id_map
    
    def _validate_interaction_vectors(self, user_id, interactions):
        """Ensure all interactions have valid video_meta_vector."""
        for inter in interactions:
            if 'video_meta_vector' not in inter:
                logger.error(f"Interaction missing 'video_meta_vector' for user {user_id}: {inter}")
                raise ValueError("All interactions must have 'video_meta_vector'")
            
            meta_vector = inter['video_meta_vector']
            try:
                meta_vector = np.asarray(meta_vector, dtype=np.float32).flatten()
                if meta_vector.shape != (1538,):
                    logger.warning(f"Reshaping video_meta_vector for user {user_id}, video {inter.get('video_id')}: "
                                f"original shape {meta_vector.shape}, expected (1538,)")
                    meta_vector = self._reshape_vector(meta_vector, 1538)
                inter['video_meta_vector'] = meta_vector
            except Exception as e:
                logger.error(f"Failed to process video_meta_vector for user {user_id}, video {inter.get('video_id')}: {e}")
                inter['video_meta_vector'] = np.zeros(1538, dtype=np.float32)
    
    def _reshape_vector(self, vector, target_size):
        """Reshape a vector to the target size by padding or truncating."""
        logger.debug(f"Reshaping vector from size {vector.size} to {target_size}")
        if vector.size < target_size:
            logger.debug(f"Vector size {vector.size} is smaller than target size {target_size}, padding with zeros")
            return np.pad(vector, (0, target_size - vector.size), mode='constant')
        elif vector.size > target_size:
            logger.debug(f"Vector size {vector.size} is larger than target size {target_size}, truncating")
            return vector[:target_size]
        return vector.reshape(target_size)
    
    def _ensure_embeddings(self, user_id, interactions):
        """Ensure all interactions have video embeddings and interaction vectors."""
        for inter in interactions:
            if 'video_embedding' not in inter:
                video_id = inter.get('video_id', inter.get('pid', None))
                if video_id:
                    try:
                        inter['video_embedding'] = self.video_feature_manager.get_video_embedding(video_id)
                    except Exception as e:
                        logger.error(f"Error fetching embedding for {video_id}: {e}")
                        inter['video_embedding'] = np.zeros(256, dtype=np.float32)
                else:
                    logger.warning(f"Skipping interaction for user {user_id}: no video_id or pid")
                    continue
            if 'interaction_vector' not in inter:
                logger.warning(f"Missing 'interaction_vector' for user {user_id}: {inter}")
                inter['interaction_vector'] = np.zeros(12, dtype=np.float32)
    
    def _create_padded_features(self, history):
        """Create padded feature matrices for sequence modeling with mean padding."""
        # Add debug logging
        logger.debug(f"Creating padded features from history with length {len(history)}")
        
        if len(history) == 0:
            logger.warning("Empty history passed to _create_padded_features")
            
        # Debug first item in history if available
        if history and len(history) > 0:
            first_item = history[0]
            logger.debug(f"First history item keys: {list(first_item.keys())}")
            if 'video_embedding' in first_item:
                logger.debug(f"First item video_embedding shape: {np.array(first_item['video_embedding']).shape}, "
                            f"non-zero: {np.count_nonzero(np.array(first_item['video_embedding']))}")
        
        video_embeddings = [inter['video_embedding'] for inter in history]
        interaction_features = [inter['interaction_vector'] for inter in history]
        video_meta_features = [inter['video_meta_vector'] for inter in history]
        
        # Calculate means (if history is empty, default to zeros)
        if history:
            video_emb_mean = np.mean(video_embeddings, axis=0, keepdims=True)
            interact_mean = np.mean(interaction_features, axis=0, keepdims=True)
            meta_mean = np.mean(video_meta_features, axis=0, keepdims=True)
            
            # Debug means
            logger.debug(f"Mean values - Video: non-zero={np.count_nonzero(video_emb_mean)}, "
                        f"Interact: non-zero={np.count_nonzero(interact_mean)}, "
                        f"Meta: non-zero={np.count_nonzero(meta_mean)}")
        else:
            video_emb_mean = np.zeros((1, 256), dtype=np.float32)
            interact_mean = np.zeros((1, 12), dtype=np.float32)
            meta_mean = np.zeros((1, 1538), dtype=np.float32)
        
        # Initialize with means instead of zeros
        padded_video_emb = np.tile(
            video_emb_mean, 
            (self.user_embedding_model.max_sequence_length, 1)
        )
        padded_interact = np.tile(
            interact_mean, 
            (self.user_embedding_model.max_sequence_length, 1)
        )
        padded_meta = np.tile(
            meta_mean, 
            (self.user_embedding_model.max_sequence_length, 1)
        )
        
        # Fill in the actual values
        seq_len = min(len(history), self.user_embedding_model.max_sequence_length)
        if seq_len > 0:
            padded_video_emb[-seq_len:] = video_embeddings[-seq_len:]
            padded_interact[-seq_len:] = interaction_features[-seq_len:]
            padded_meta[-seq_len:] = video_meta_features[-seq_len:]
            
            # Debug final padded arrays
            logger.debug(f"Final padded arrays - Video: non-zero={np.count_nonzero(padded_video_emb)}, "
                        f"Interact: non-zero={np.count_nonzero(padded_interact)}, "
                        f"Meta: non-zero={np.count_nonzero(padded_meta)}")
        else:
            logger.warning("Zero sequence length in _create_padded_features")
        
        # Flatten for model input
        flat_video_emb = padded_video_emb.flatten()
        flat_interact = padded_interact.flatten()
        flat_meta = padded_meta.flatten()
        
        logger.debug(f"Returning sequence length: {seq_len}")
        return flat_video_emb, flat_interact, flat_meta, seq_len
    
    def _map_ids(self, user_id, video_id, user_id_map, video_id_map, split_name):
        """Map string IDs to integer IDs using the provided mappings."""
        str_user_id = str(user_id)
        str_video_id = str(video_id)
        
        mapped_user_id = user_id_map.get(str_user_id)
        mapped_video_id = video_id_map.get(str_video_id)
        
        if mapped_user_id is None:
            logger.error(f"Failed to map user_id {user_id} (str: {str_user_id}) for {split_name}")
            logger.debug(f"Available user_id keys: {list(user_id_map.keys())[:10]}...")
            return None, None
        
        if mapped_video_id is None:
            logger.error(f"Failed to map video_id {video_id} (str: {str_video_id}) for user {user_id} in {split_name}")
            logger.debug(f"Available video_id keys: {list(video_id_map.keys())[:10]}...")
            return None, None
        
        logger.debug(f"ID Mapping for {split_name}: User {user_id} -> {mapped_user_id}, Video {video_id} -> {mapped_video_id}")
        return mapped_user_id, mapped_video_id
    
    def _save_batch(self, processed_data, output_dir, batch_idx, split_name):
        """Save a batch of processed data to a Parquet file."""
        df_batch = pd.DataFrame(processed_data)
        unique_users = df_batch['user_id'].nunique()
        unique_videos = df_batch['video_id'].nunique()
        
        # Debug: Print ID ranges in batch
        logger.debug(f"Batch {batch_idx} ID ranges - Users: [{df_batch['user_id'].min()}, {df_batch['user_id'].max()}], "
                    f"Videos: [{df_batch['video_id'].min()}, {df_batch['video_id'].max()}]")
        
        logger.info(f"{'Final batch' if batch_idx == 0 else 'Batch'} {batch_idx} for {split_name}: "
                    f"{len(df_batch)} rows, {unique_users} unique users, {unique_videos} unique videos")
        
        # Check for NaN IDs
        nan_user_ids = df_batch['user_id'].isna().sum()
        nan_video_ids = df_batch['video_id'].isna().sum()
        
        if nan_user_ids > 0 or nan_video_ids > 0:
            logger.error(f"NaN IDs in batch {batch_idx} for {split_name}: "
                    f"NaN user_ids: {nan_user_ids}, NaN video_ids: {nan_video_ids}")
        
        # Debug: Save sample of the data
        if batch_idx == 0:  # For the first batch, save a sample
            sample_df = df_batch.head(10).copy()
            sample_df.to_csv(os.path.join(output_dir, f"sample_batch_{batch_idx}.csv"))
            logger.debug(f"Saved sample of batch {batch_idx} to CSV for inspection")
        
        df_batch.to_parquet(os.path.join(output_dir, f"batch_{batch_idx}.parquet"))
        self.clear_caches()
        gc.collect()
    
    def _validate_processed_files(self, output_dir, user_id_map, video_id_map, split_name):
        """Validate the processed Parquet files and log statistics."""
        parquet_files = glob.glob(os.path.join(output_dir, "*.parquet"))
        total_rows = 0
        all_user_ids = set()
        all_video_ids = set()
        
        for file in parquet_files:
            df = pd.read_parquet(file)
            total_rows += len(df)
            all_user_ids.update(df['user_id'].unique())
            all_video_ids.update(df['video_id'].unique())
            
            # Check for invalid IDs
            invalid_users = df[(df['user_id'].isna()) | (df['user_id'] < 1) | (df['user_id'] > len(user_id_map))]
            invalid_videos = df[(df['video_id'].isna()) | (df['video_id'] < 1) | (df['video_id'] > len(video_id_map))]
            
            if not invalid_users.empty:
                logger.error(f"Invalid user IDs in {file}: {invalid_users['user_id'].tolist()}")
            if not invalid_videos.empty:
                logger.error(f"Invalid video IDs in {file}: {invalid_videos['video_id'].tolist()}")
            
            # Debug: Check ID distribution
            logger.debug(f"File {file} user ID range: [{df['user_id'].min()}, {df['user_id'].max()}]")
            logger.debug(f"File {file} video ID range: [{df['video_id'].min()}, {df['video_id'].max()}]")
        
        # Log statistics
        logger.info(f"{split_name.capitalize()} data saved to {output_dir}: {total_rows} rows, "
                    f"{len(all_user_ids)} unique users, {len(all_video_ids)} unique videos")
        logger.info(f"Expected users: {len(user_id_map)}, Expected videos: {len(video_id_map)}")
        
        # Debug: Check for missing IDs
        expected_user_ids = set(user_id_map.values())
        expected_video_ids = set(video_id_map.values())
        missing_user_ids = expected_user_ids - all_user_ids
        missing_video_ids = expected_video_ids - all_video_ids
        
        if missing_user_ids:
            logger.warning(f"Missing user IDs in {split_name}: {len(missing_user_ids)} IDs not found in processed data")
            logger.debug(f"First 10 missing user IDs: {list(missing_user_ids)[:10]}")
        
        if missing_video_ids:
            logger.warning(f"Missing video IDs in {split_name}: {len(missing_video_ids)} IDs not found in processed data")
            logger.debug(f"First 10 missing video IDs: {list(missing_video_ids)[:10]}")
        
    def _process_all_splits(self, interaction_dict, batch_size, user_id_map, video_id_map):
        """
        Process all splits with time-aware splitting while preventing data leakage.
        Ensures no (user, video) pair appears in multiple splits.
        """
        # Define a helper function to normalize timestamp values
        def normalize_timestamps(interactions):
            """Ensure all timestamp values are integers"""
            for interaction in interactions:
                for field in ['exposed_time', 'timestamp']:
                    if field in interaction:
                        if isinstance(interaction[field], str):
                            try:
                                interaction[field] = int(interaction[field])
                            except (ValueError, TypeError):
                                interaction[field] = 0
            return interactions
        
        # Track unique (user, video) pairs to prevent leakage
        seen_user_video_pairs = set()
        unique_interactions = []
        duplicate_count = 0
        users_with_few_interactions = []
        min_interactions_required = 5
        
        # First pass: collect unique (user, video) pairs with earliest timestamp
        user_video_first_interaction = {}
        
        for user_id, interactions in interaction_dict.items():
            if len(interactions) < min_interactions_required:
                users_with_few_interactions.append(user_id)
                continue
            
            # Normalize timestamps
            normalized_interactions = normalize_timestamps(interactions)
            
            # Group by video_id and keep only the earliest interaction
            for interaction in normalized_interactions:
                video_id = interaction.get('video_id', interaction.get('pid'))
                if not video_id:
                    continue
                    
                pair_key = (user_id, video_id)
                
                # Get timestamp
                timestamp = interaction.get('exposed_time', interaction.get('timestamp', 0))
                if isinstance(timestamp, str):
                    try:
                        timestamp = int(float(timestamp))
                    except (ValueError, TypeError):
                        timestamp = 0
                
                # Keep only the first interaction for each (user, video) pair
                if pair_key not in user_video_first_interaction:
                    user_video_first_interaction[pair_key] = {
                        **interaction,
                        'user_id': user_id,
                        'timestamp': timestamp
                    }
                else:
                    # If we already have this pair, keep the one with earlier timestamp
                    existing_timestamp = user_video_first_interaction[pair_key]['timestamp']
                    if timestamp < existing_timestamp:
                        user_video_first_interaction[pair_key] = {
                            **interaction,
                            'user_id': user_id,
                            'timestamp': timestamp
                        }
                    duplicate_count += 1
        
        logger.info(f"Removed {duplicate_count} duplicate (user, video) interactions")
        logger.info(f"Filtered out {len(users_with_few_interactions)} users with fewer than {min_interactions_required} interactions")
        
        # Convert to list and separate cold-start users
        cold_start_interactions = []
        regular_interactions = []
        
        for interaction in user_video_first_interaction.values():
            user_id = interaction['user_id']
            if isinstance(user_id, str) and user_id.startswith("cold_user_"):
                cold_start_interactions.append(interaction)
            else:
                regular_interactions.append(interaction)
        
        logger.info(f"Processing {len(regular_interactions)} unique regular interactions and {len(cold_start_interactions)} cold-start interactions")
        
        # Sort regular interactions by timestamp
        if regular_interactions:
            # Check timestamp field availability
            if 'exposed_time' in regular_interactions[0]:
                timestamp_field = 'exposed_time'
            elif 'timestamp' in regular_interactions[0]:
                timestamp_field = 'timestamp'
            else:
                timestamp_field = None
                logger.warning("No timestamp field found, using random order for splitting")
            
            if timestamp_field:
                def get_timestamp(interaction):
                    return interaction.get('timestamp', 0)
                
                try:
                    regular_interactions.sort(key=get_timestamp)
                    if len(regular_interactions) >= 2:
                        first_ts = regular_interactions[0]['timestamp']
                        last_ts = regular_interactions[-1]['timestamp']
                        logger.info(f"Sorted interactions: first timestamp {first_ts}, last timestamp {last_ts}")
                except Exception as e:
                    logger.error(f"Error sorting interactions: {str(e)}")
                    random.shuffle(regular_interactions)
            else:
                random.shuffle(regular_interactions)
        
        # Calculate split indices
        total_regular = len(regular_interactions)
        train_ratio = 0.8
        valid_ratio = 0.1
        
        train_end = int(total_regular * train_ratio)
        valid_end = train_end + int(total_regular * valid_ratio)
        
        # Split regular interactions by time
        train_interactions = regular_interactions[:train_end]
        valid_interactions = regular_interactions[train_end:valid_end]
        test_interactions = regular_interactions[valid_end:]
        
        # Verify no overlap in (user, video) pairs
        train_pairs = {(i['user_id'], i.get('video_id', i.get('pid'))) for i in train_interactions}
        valid_pairs = {(i['user_id'], i.get('video_id', i.get('pid'))) for i in valid_interactions}
        test_pairs = {(i['user_id'], i.get('video_id', i.get('pid'))) for i in test_interactions}
        
        train_valid_overlap = train_pairs & valid_pairs
        train_test_overlap = train_pairs & test_pairs
        valid_test_overlap = valid_pairs & test_pairs
        
        if train_valid_overlap:
            logger.error(f"DATA LEAKAGE: {len(train_valid_overlap)} (user,video) pairs in both train and valid!")
        if train_test_overlap:
            logger.error(f"DATA LEAKAGE: {len(train_test_overlap)} (user,video) pairs in both train and test!")
        if valid_test_overlap:
            logger.error(f"DATA LEAKAGE: {len(valid_test_overlap)} (user,video) pairs in both valid and test!")
        
        # Handle cold-start users
        if cold_start_interactions:
            # Group cold-start interactions by user
            cold_user_dict = {}
            for interaction in cold_start_interactions:
                user_id = interaction['user_id']
                if user_id not in cold_user_dict:
                    cold_user_dict[user_id] = []
                cold_user_dict[user_id].append(interaction)
            
            # Distribute cold users
            cold_users = list(cold_user_dict.keys())
            random.shuffle(cold_users)
            
            cold_train_users = cold_users[:int(len(cold_users) * 0.6)]
            cold_valid_users = cold_users[int(len(cold_users) * 0.6):int(len(cold_users) * 0.8)]
            cold_test_users = cold_users[int(len(cold_users) * 0.8):]
            
            for user_id in cold_train_users:
                train_interactions.extend(cold_user_dict[user_id])
            for user_id in cold_valid_users:
                valid_interactions.extend(cold_user_dict[user_id])
            for user_id in cold_test_users:
                test_interactions.extend(cold_user_dict[user_id])
            
            logger.info(f"Distributed cold-start users: {len(cold_train_users)} to train, "
                    f"{len(cold_valid_users)} to valid, {len(cold_test_users)} to test")
        
        # Log statistics
        total_interactions = len(train_interactions) + len(valid_interactions) + len(test_interactions)
        logger.info(f"Time-aware split - Total: {total_interactions} unique interactions")
        logger.info(f"Train: {len(train_interactions)} interactions ({len(train_interactions)/total_interactions:.2%})")
        logger.info(f"Valid: {len(valid_interactions)} interactions ({len(valid_interactions)/total_interactions:.2%})")
        logger.info(f"Test: {len(test_interactions)} interactions ({len(test_interactions)/total_interactions:.2%})")
        
        # Log timestamp ranges
        if regular_interactions and 'timestamp' in regular_interactions[0]:
            for split_name, split_data in [("Train", train_interactions), 
                                        ("Valid", valid_interactions), 
                                        ("Test", test_interactions)]:
                if split_data:
                    timestamps = [i['timestamp'] for i in split_data if 'timestamp' in i and i['timestamp'] > 0]
                    if timestamps:
                        try:
                            min_ts = min(timestamps)
                            max_ts = max(timestamps)
                            min_time = datetime.fromtimestamp(min_ts)
                            max_time = datetime.fromtimestamp(max_ts)
                            logger.info(f"{split_name} time range: {min_time} to {max_time}")
                        except Exception as e:
                            logger.error(f"Error calculating {split_name} time range: {str(e)}")
        
        # Convert back to user-based dictionaries
        train_dict = self._interactions_to_user_dict(train_interactions)
        valid_dict = self._interactions_to_user_dict(valid_interactions) 
        test_dict = self._interactions_to_user_dict(test_interactions)
        
        # Ensure minimum interactions per user after deduplication
        initial_sizes = (len(train_dict), len(valid_dict), len(test_dict))
        train_dict = self._ensure_minimum_interactions(train_dict, min_interactions=5)
        valid_dict = self._ensure_minimum_interactions(valid_dict, min_interactions=5)
        test_dict = self._ensure_minimum_interactions(test_dict, min_interactions=5)
        final_sizes = (len(train_dict), len(valid_dict), len(test_dict))
        
        if initial_sizes != final_sizes:
            logger.info(f"Users filtered due to insufficient interactions after deduplication: "
                    f"Train: {initial_sizes[0] - final_sizes[0]}, "
                    f"Valid: {initial_sizes[1] - final_sizes[1]}, "
                    f"Test: {initial_sizes[2] - final_sizes[2]}")
        
        # Log final user statistics
        logger.info(f"Final user counts - Train: {len(train_dict)}, Valid: {len(valid_dict)}, Test: {len(test_dict)}")
        
        # Final overlap check
        self._log_user_overlap_statistics(train_dict, valid_dict, test_dict)
        
        # Process each split
        train_dir, _, _ = self.process_interactions(
            train_dict, 'train', batch_size=batch_size, output_dir="processed_data_train",
            user_id_map=user_id_map, video_id_map=video_id_map
        )
        valid_dir, _, _ = self.process_interactions(
            valid_dict, 'valid', batch_size=batch_size, output_dir="processed_data_valid",
            user_id_map=user_id_map, video_id_map=video_id_map
        )
        test_dir, _, _ = self.process_interactions(
            test_dict, 'test', batch_size=batch_size, output_dir="processed_data_test",
            user_id_map=user_id_map, video_id_map=video_id_map
        )
        
        # Final verification
        self._check_user_overlap(train_dir, valid_dir, test_dir)
        
        return train_dir, valid_dir, test_dir

    def _ensure_minimum_interactions(self, user_dict, min_interactions=5):
        """
        Ensure each user in the dictionary has at least min_interactions.
        Remove users who don't meet this requirement.
        """
        filtered_dict = {}
        removed_users = []
        
        for user_id, interactions in user_dict.items():
            if len(interactions) >= min_interactions:
                filtered_dict[user_id] = interactions
            else:
                removed_users.append(user_id)
        
        if removed_users:
            logger.info(f"Removed {len(removed_users)} users with fewer than {min_interactions} interactions")
        
        return filtered_dict

    def _interactions_to_user_dict(self, interactions):
        """
        Convert a flat list of interactions to a user-based dictionary.
        Each interaction must have a 'user_id' field.
        Preserves chronological order within each user's interactions.
        """
        user_dict = {}
        for interaction in interactions:
            user_id = interaction['user_id']
            if user_id not in user_dict:
                user_dict[user_id] = []
            
            # Make a copy to avoid modifying the original
            interaction_copy = interaction.copy()
            user_dict[user_id].append(interaction_copy)
        
        # Sort each user's interactions by timestamp if available
        users_with_timestamps = 0
        for user_id, user_interactions in user_dict.items():
            if user_interactions and ('exposed_time' in user_interactions[0] or 'timestamp' in user_interactions[0]):
                users_with_timestamps += 1
                # Determine timestamp field
                timestamp_field = 'exposed_time' if 'exposed_time' in user_interactions[0] else 'timestamp'
                
                # Define a safe conversion function for timestamps
                def safe_timestamp(interaction):
                    timestamp_value = interaction.get(timestamp_field, 0)
                    if isinstance(timestamp_value, str):
                        try:
                            return int(timestamp_value)
                        except (ValueError, TypeError):
                            return 0
                    return timestamp_value
                
                # Sort by timestamp in ascending order
                try:
                    user_dict[user_id] = sorted(user_interactions, key=safe_timestamp)
                    # Debug: Check timestamp sorting
                    if len(user_interactions) > 1:
                        first_ts = safe_timestamp(user_dict[user_id][0])
                        last_ts = safe_timestamp(user_dict[user_id][-1])
                        logger.debug(f"User {user_id}: Sorted interactions from timestamp {first_ts} to {last_ts}")
                except Exception as e:
                    logger.error(f"Error sorting interactions for user {user_id}: {str(e)}")
        
        logger.debug(f"Sorted interactions by timestamp for {users_with_timestamps}/{len(user_dict)} users")
        
        # Log interaction distribution
        interaction_counts = {user_id: len(interactions) for user_id, interactions in user_dict.items()}
        if interaction_counts:
            min_count = min(interaction_counts.values())
            max_count = max(interaction_counts.values())
            avg_count = sum(interaction_counts.values()) / len(interaction_counts)
            logger.debug(f"User interaction counts: Min={min_count}, Max={max_count}, Avg={avg_count:.2f}")
            
            # Log sample of users with low interaction counts
            low_count_users = [uid for uid, count in interaction_counts.items() if count < 5]
            if low_count_users:
                logger.debug(f"Found {len(low_count_users)} users with <5 interactions. Sample: {low_count_users[:5]}")
        
        # Filter out users with too few interactions for validation/testing
        min_interactions = 5  # Minimum required for validation/testing
        filtered_user_dict = {
            user_id: interactions for user_id, interactions in user_dict.items() 
            if len(interactions) >= min_interactions
        }
        
        filtered_count = len(user_dict) - len(filtered_user_dict)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} users with fewer than {min_interactions} interactions from split")
        
        return filtered_user_dict

    def _log_user_overlap_statistics(self, train_dict, valid_dict, test_dict):
        """
        Log statistics about user overlap between splits.
        """
        train_users = set(train_dict.keys())
        valid_users = set(valid_dict.keys())
        test_users = set(test_dict.keys())
        
        # Check overlaps
        train_valid_overlap = train_users.intersection(valid_users)
        train_test_overlap = train_users.intersection(test_users)
        valid_test_overlap = valid_users.intersection(test_users)
        all_splits_overlap = train_users.intersection(valid_users).intersection(test_users)
        
        logger.info(f"User overlap statistics:")
        logger.info(f"Users in train and valid: {len(train_valid_overlap)} ({len(train_valid_overlap)/len(train_users):.2%} of train users)")
        logger.info(f"Users in train and test: {len(train_test_overlap)} ({len(train_test_overlap)/len(train_users):.2%} of train users)")
        logger.info(f"Users in valid and test: {len(valid_test_overlap)} ({len(valid_test_overlap)/len(valid_users):.2%} of valid users)")
        logger.info(f"Users in all three splits: {len(all_splits_overlap)}")
    def _check_user_overlap(self, train_dir, valid_dir, test_dir):
        """
        Check user overlap between splits to verify time-aware split-by-ratio is working.
        """
        train_users = set()
        valid_users = set()
        test_users = set()
        
        # Collect users from each split
        for file in glob.glob(os.path.join(train_dir, "*.parquet")):
            df = pd.read_parquet(file)
            train_users.update(df['user_id'].unique())
        
        for file in glob.glob(os.path.join(valid_dir, "*.parquet")):
            df = pd.read_parquet(file)
            valid_users.update(df['user_id'].unique())
        
        for file in glob.glob(os.path.join(test_dir, "*.parquet")):
            df = pd.read_parquet(file)
            test_users.update(df['user_id'].unique())
        
        # Calculate overlaps
        train_valid_overlap = train_users.intersection(valid_users)
        train_test_overlap = train_users.intersection(test_users)
        valid_test_overlap = valid_users.intersection(test_users)
        all_overlap = train_users.intersection(valid_users).intersection(test_users)
        
        # Log overlap statistics with safe division
        logger.info("User overlap statistics (time-aware split-by-ratio):")
        
        # Safe calculation for train-valid overlap
        if train_users and valid_users:  # Check if both sets are non-empty
            train_valid_pct = len(train_valid_overlap) / len(train_users) * 100
            logger.info(f"Train-Valid overlap: {len(train_valid_overlap)} users ({train_valid_pct:.1f}%)")
        else:
            logger.warning("Train-Valid overlap: Cannot calculate percentage (no users in valid set)")
        
        # Safe calculation for train-test overlap
        if train_users and test_users:  # Check if both sets are non-empty
            train_test_pct = len(train_test_overlap) / len(train_users) * 100
            logger.info(f"Train-Test overlap: {len(train_test_overlap)} users ({train_test_pct:.1f}%)")
        else:
            logger.warning("Train-Test overlap: Cannot calculate percentage (no users in test set)")
        
        # Safe calculation for valid-test overlap
        if valid_users and test_users:  # Check if both sets are non-empty
            valid_test_pct = len(valid_test_overlap) / len(valid_users) * 100
            logger.info(f"Valid-Test overlap: {len(valid_test_overlap)} users ({valid_test_pct:.1f}%)")
        else:
            logger.warning("Valid-Test overlap: Cannot calculate percentage (empty valid or test set)")
        
        if train_users and valid_users and test_users:
            logger.info(f"Users in all splits: {len(all_overlap)} users")
        else:
            logger.warning("Users in all splits: Cannot calculate (one or more splits are empty)")
        
        logger.info(f"Total unique users - Train: {len(train_users)}, Valid: {len(valid_users)}, Test: {len(test_users)}")
        
        # Check if validation and test sets are empty and report the issue
        if not valid_users or not test_users:
            logger.error("ERROR: Empty validation or test sets detected. This will cause training failures.")
            logger.error("The most likely cause is that the history/target splitting logic in process_interactions is not creating valid examples.")
            logger.error("Check your implementation of history/target handling in the process_interactions method.")
    
    def load_or_process_data(self, interaction_dict, batch_size, user_id_map=None, video_id_map=None, 
                         force_reprocess=False, processed_data_dir=None):
            """
            Loads pre-processed data if available, or processes data from scratch if not.
            
            Args:
                interaction_dict: Dictionary of user interactions
                batch_size: Batch size for processing
                user_id_map: Optional mapping of user IDs to integers
                video_id_map: Optional mapping of video IDs to integers
                force_reprocess: If True, reprocess data even if processed data exists
                processed_data_dir: Optional directory containing processed_data_train, processed_data_valid, etc.
                
            Returns:
                Tuple of (train_dir, valid_dir, test_dir, user_id_map, video_id_map)
            """
            if processed_data_dir is None:
                processed_data_dir = "."  # Use current directory
                
            train_dir = os.path.join(processed_data_dir, "processed_data_train")
            valid_dir = os.path.join(processed_data_dir, "processed_data_valid")
            test_dir = os.path.join(processed_data_dir, "processed_data_test")
            
            # Check if processed data exists and is valid
            data_exists = all(os.path.exists(dir_path) for dir_path in [train_dir, valid_dir, test_dir])
            
            if data_exists and not force_reprocess:
                # Check if data files exist in each directory
                train_files = glob.glob(os.path.join(train_dir, "*.parquet"))
                valid_files = glob.glob(os.path.join(valid_dir, "*.parquet"))
                test_files = glob.glob(os.path.join(test_dir, "*.parquet"))
                
                if train_files and (valid_files or test_files):  # At least train and one of valid/test must exist
                    logger.info("Found existing processed data. Loading instead of reprocessing.")
                    
                    # Load user_id_map and video_id_map if they exist
                    id_maps_dir = os.path.join(processed_data_dir, "data")
                    user_map_path = os.path.join(id_maps_dir, "user_id_map.json")
                    video_map_path = os.path.join(id_maps_dir, "video_id_map.json")
                    
                    if os.path.exists(user_map_path) and os.path.exists(video_map_path):
                        with open(user_map_path, 'r') as f:
                            user_id_map = json.load(f)
                        with open(video_map_path, 'r') as f:
                            video_id_map = json.load(f)
                        logger.info(f"Loaded ID mappings: {len(user_id_map)} users, {len(video_id_map)} videos")
                    else:
                        # Try to infer mappings from data if files don't exist
                        logger.warning("ID mapping files not found. Attempting to infer from data.")
                        user_id_map, video_id_map = self._infer_id_mappings_from_data(train_dir)
                    
                    # Check user overlap in loaded data
                    self._check_user_overlap(train_dir, valid_dir, test_dir)
                    
                    return train_dir, valid_dir, test_dir, user_id_map, video_id_map
                    
                else:
                    logger.warning("Found processed data directories but they seem to be empty or incomplete.")
            
            # If we get here, either no data exists, force_reprocess=True, or data was invalid
            logger.info("Processing data from scratch...")
            
            # Use the existing _process_all_splits method
            train_dir, valid_dir, test_dir = self._process_all_splits(interaction_dict, batch_size, 
                                                                    user_id_map, video_id_map)
            
            # Store ID mappings for future use
            data_dir = os.path.join(processed_data_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            
            with open(os.path.join(data_dir, "user_id_map.json"), "w") as f:
                json.dump(user_id_map, f)
            with open(os.path.join(data_dir, "video_id_map.json"), "w") as f:
                json.dump(video_id_map, f)
            
            logger.info(f"Saved ID mappings to {data_dir}")
            
            return train_dir, valid_dir, test_dir, user_id_map, video_id_map

    def _infer_id_mappings_from_data(self, train_dir):
        """
        Infer user_id_map and video_id_map from processed training data.
        """
        logger.info("Inferring ID mappings from processed data...")
        
        # Initialize empty mappings
        user_id_map = {}
        video_id_map = {}
        
        try:
            # Read all parquet files in training directory
            train_files = glob.glob(os.path.join(train_dir, "*.parquet"))
            
            # Process each file to extract user and video IDs
            for file in train_files:
                df = pd.read_parquet(file)
                
                # Extract all unique user IDs and their mapped values
                user_df = df[['user_id']].drop_duplicates()
                for _, row in user_df.iterrows():
                    # Store the integer ID as the value, and use string keys for consistency
                    original_id = str(row['user_id'])
                    user_id_map[original_id] = int(row['user_id'])
                
                # Extract all unique video IDs and their mapped values
                video_df = df[['video_id']].drop_duplicates()
                for _, row in video_df.iterrows():
                    # Store the integer ID as the value, and use string keys for consistency
                    original_id = str(row['video_id'])
                    video_id_map[original_id] = int(row['video_id'])
            
            logger.info(f"Inferred ID mappings from data: {len(user_id_map)} users, {len(video_id_map)} videos")
            
            return user_id_map, video_id_map
        
        except Exception as e:
            logger.error(f"Failed to infer ID mappings from data: {str(e)}")
            # Return empty mappings which will trigger normal processing
            return {}, {}
    def _check_data_leakage(self, train_dataset, valid_dataset, test_dataset):
        """Check for potential data leakage between splits - COMPREHENSIVE VERSION."""
        logger.info("🔍 Checking for data leakage across ALL data...")
        
        try:
            def extract_all_pairs_from_dataset(dataset, dataset_name):
                """Extract ALL (user_id, video_id) pairs from entire dataset"""
                if dataset is None or dataset.num_rows == 0:
                    logger.warning(f"⚠️  {dataset_name} dataset is empty")
                    return set()
                    
                logger.info(f"Extracting ALL pairs from {dataset_name} ({dataset.num_rows:,} total rows)")
                
                # Get the entire dataset, not just a sample
                dataset_ddf = dataset.to_ddf()
                
                # Process the ENTIRE dataset - this is the key fix
                if hasattr(dataset_ddf, 'compute'):
                    # Dask DataFrame - compute entire dataset
                    dataset_df = dataset_ddf.compute()
                else:
                    dataset_df = dataset_ddf
                    
                # Convert to pandas if needed
                if hasattr(dataset_df, 'to_pandas'):
                    # cuDF DataFrame - convert to pandas
                    dataset_df = dataset_df.to_pandas()
                
                # Extract ALL (user_id, video_id) pairs
                all_pairs = set(zip(dataset_df['user_id'].tolist(), dataset_df['video_id'].tolist()))
                
                logger.info(f"✅ Extracted {len(all_pairs):,} unique (user,video) pairs from {dataset_name}")
                return all_pairs
            
            # Extract ALL pairs from each dataset (not samples!)
            train_pairs = extract_all_pairs_from_dataset(train_dataset, "TRAIN")
            
            leakage_detected = False
            
            if valid_dataset and valid_dataset.num_rows > 0:
                valid_pairs = extract_all_pairs_from_dataset(valid_dataset, "VALID")
                
                # Check for identical (user_id, video_id) pairs between train and valid
                train_valid_overlap = train_pairs.intersection(valid_pairs)
                if train_valid_overlap:
                    logger.error(f"❌ TRAIN-VALID DATA LEAKAGE DETECTED: {len(train_valid_overlap):,} identical (user,video) pairs!")
                    logger.error(f"   Sample overlapping pairs: {list(train_valid_overlap)[:5]}")
                    leakage_detected = True
                else:
                    logger.info("✅ No identical (user,video) pairs between TRAIN-VALID")
                    
                # Check test dataset if it exists
                if test_dataset and test_dataset.num_rows > 0:
                    test_pairs = extract_all_pairs_from_dataset(test_dataset, "TEST")
                    
                    # Check train-test overlap
                    train_test_overlap = train_pairs.intersection(test_pairs)
                    if train_test_overlap:
                        logger.error(f"❌ TRAIN-TEST DATA LEAKAGE DETECTED: {len(train_test_overlap):,} identical (user,video) pairs!")
                        logger.error(f"   Sample overlapping pairs: {list(train_test_overlap)[:5]}")
                        leakage_detected = True
                    else:
                        logger.info("✅ No identical (user,video) pairs between TRAIN-TEST")
                    
                    # Check valid-test overlap  
                    valid_test_overlap = valid_pairs.intersection(test_pairs)
                    if valid_test_overlap:
                        logger.error(f"❌ VALID-TEST DATA LEAKAGE DETECTED: {len(valid_test_overlap):,} identical (user,video) pairs!")
                        logger.error(f"   Sample overlapping pairs: {list(valid_test_overlap)[:5]}")
                        leakage_detected = True
                    else:
                        logger.info("✅ No identical (user,video) pairs between VALID-TEST")
                
                # Report statistics on user overlap (this is expected in time-aware splits)
                train_users = set()
                valid_users = set()
                
                # Extract users from the pairs we already have
                for user_id, video_id in train_pairs:
                    train_users.add(user_id)
                for user_id, video_id in valid_pairs:
                    valid_users.add(user_id)
                    
                user_overlap = train_users.intersection(valid_users)
                user_overlap_pct = len(user_overlap) / len(train_users) * 100 if train_users else 0
                
                logger.info(f"📊 User overlap: {len(user_overlap):,} users ({user_overlap_pct:.1f}% of train users)")
                if user_overlap_pct > 50:
                    logger.info("   ℹ️  High user overlap is expected in time-aware splits")
                
                # Summary statistics
                if test_dataset and test_dataset.num_rows > 0:
                    test_pairs = extract_all_pairs_from_dataset(test_dataset, "TEST")
                    total_unique_pairs = len(train_pairs.union(valid_pairs).union(test_pairs))
                    total_pairs_sum = len(train_pairs) + len(valid_pairs) + len(test_pairs)
                else:
                    total_unique_pairs = len(train_pairs.union(valid_pairs))
                    total_pairs_sum = len(train_pairs) + len(valid_pairs)
                
                overlap_count = total_pairs_sum - total_unique_pairs
                
                logger.info(f"📊 COMPREHENSIVE LEAKAGE SUMMARY:")
                logger.info(f"   Train pairs: {len(train_pairs):,}")
                logger.info(f"   Valid pairs: {len(valid_pairs):,}")
                if test_dataset and test_dataset.num_rows > 0:
                    logger.info(f"   Test pairs:  {len(test_pairs):,}")
                logger.info(f"   Total pairs (if no overlap): {total_pairs_sum:,}")
                logger.info(f"   Actual unique pairs: {total_unique_pairs:,}")
                logger.info(f"   Overlapping pairs: {overlap_count:,}")
                
            else:
                logger.warning("⚠️  No validation dataset to check against")
                
            if leakage_detected:
                logger.error("❌ DATA LEAKAGE FOUND! Your evaluation metrics are unreliable.")
                logger.error("   You need to fix your data splitting logic before trusting results.")
                return False
            else:
                logger.info("✅ NO DATA LEAKAGE DETECTED - Your splits are clean!")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error during comprehensive data leakage check: {str(e)}")
            logger.error(f"   Error type: {type(e).__name__}")
            import traceback
            logger.debug(f"   Full traceback: {traceback.format_exc()}")
            return False

    def _validate_id_mappings(self):
        """Validate ID mappings for consistency."""
        logger.info("🔍 Validating ID mappings...")
        
        try:
            if hasattr(self, 'user_id_map') and hasattr(self, 'video_id_map'):
                # Check for gaps in ID mapping
                user_ids = list(self.user_id_map.values())
                video_ids = list(self.video_id_map.values())
                
                # IDs should be continuous from 1 to max
                expected_user_ids = set(range(1, len(user_ids) + 1))
                expected_video_ids = set(range(1, len(video_ids) + 1))
                
                actual_user_ids = set(user_ids)
                actual_video_ids = set(video_ids)
                
                # Check user IDs
                if actual_user_ids != expected_user_ids:
                    missing_users = expected_user_ids - actual_user_ids
                    extra_users = actual_user_ids - expected_user_ids
                    logger.error(f"❌ User ID mapping issues:")
                    if missing_users:
                        logger.error(f"   Missing IDs: {sorted(list(missing_users))[:10]}...")
                    if extra_users:
                        logger.error(f"   Extra IDs: {sorted(list(extra_users))[:10]}...")
                else:
                    logger.info(f"✅ User ID mapping is continuous (1 to {len(user_ids)})")
                    
                # Check video IDs
                if actual_video_ids != expected_video_ids:
                    missing_videos = expected_video_ids - actual_video_ids
                    extra_videos = actual_video_ids - expected_video_ids
                    logger.error(f"❌ Video ID mapping issues:")
                    if missing_videos:
                        logger.error(f"   Missing IDs: {sorted(list(missing_videos))[:10]}...")
                    if extra_videos:
                        logger.error(f"   Extra IDs: {sorted(list(extra_videos))[:10]}...")
                else:
                    logger.info(f"✅ Video ID mapping is continuous (1 to {len(video_ids)})")
                    
                # Check for duplicates
                if len(user_ids) != len(set(user_ids)):
                    logger.error("❌ Duplicate values in user_id_map")
                
                if len(video_ids) != len(set(video_ids)):
                    logger.error("❌ Duplicate values in video_id_map")
                    
            else:
                logger.warning("⚠️  ID mappings not found (user_id_map or video_id_map missing)")
                
        except Exception as e:
            logger.error(f"❌ Error validating ID mappings: {str(e)}")

    def _validate_embeddings(self):
        """Validate embedding quality."""
        logger.info("🔍 Validating embeddings...")
        
        try:
            # Check if embeddings are not all zeros
            sample_embeddings = []
            cache_items = list(embedding_cache.items())[:10] if embedding_cache else []
            
            if not cache_items:
                logger.warning("⚠️  No embeddings found in cache")
                return
                
            for video_id, embedding in cache_items:
                if embedding is not None and hasattr(embedding, 'shape'):
                    sample_embeddings.append(embedding)
            
            if sample_embeddings:
                norms = [np.linalg.norm(emb) for emb in sample_embeddings]
                mean_norm = np.mean(norms)
                std_norm = np.std(norms)
                
                logger.info(f"📊 Embedding statistics:")
                logger.info(f"   Mean norm: {mean_norm:.3f}")
                logger.info(f"   Std norm: {std_norm:.3f}")
                logger.info(f"   Min norm: {min(norms):.3f}")
                logger.info(f"   Max norm: {max(norms):.3f}")
                
                if mean_norm < 0.1:
                    logger.error("❌ Embeddings have very low norms (possibly zeros or near-zeros)")
                elif mean_norm > 100:
                    logger.warning("⚠️  Embeddings have very high norms (may need normalization)")
                else:
                    logger.info("✅ Embedding norms look healthy")
                    
                # Check for all-zero embeddings
                zero_count = sum(1 for emb in sample_embeddings if np.allclose(emb, 0))
                if zero_count > 0:
                    logger.warning(f"⚠️  Found {zero_count}/{len(sample_embeddings)} zero embeddings in sample")
                    
            else:
                logger.warning("⚠️  No valid embeddings found to validate")
                
        except Exception as e:
            logger.error(f"❌ Error validating embeddings: {str(e)}")

    def _analyze_training_metrics(self):
        """Analyze training metrics for red flags."""
        logger.info("🔍 Analyzing training metrics...")
        
        # Guidelines for healthy metrics
        logger.info("📋 Healthy Metrics Guidelines:")
        logger.info("   • Train-Valid NDCG gap should be < 0.15")
        logger.info("   • Train-Valid Recall gap should be < 0.20") 
        logger.info("   • NDCG@10 should be 0.20-0.80 range")
        logger.info("   • Recall@10 should be 0.30-0.90 range")
        logger.info("   • Loss should decrease smoothly")
        logger.info("   • No NaN or inf values")
        logger.info("   • Training metrics shouldn't be 'too perfect' (>95%)")
        
        # Red flags to watch for
        logger.info("🚨 Red Flags to Watch For:")
        logger.info("   • Perfect training metrics (>98% recall)")
        logger.info("   • Large train-validation gaps (>0.20)")
        logger.info("   • Validation metrics not improving")
        logger.info("   • Sudden metric jumps or drops")
        
    def validate_training_results(self, train_dataset, valid_dataset, test_dataset):
        """
        Comprehensive validation of training results and data quality.
        """
        logger.info("=" * 50)
        logger.info("🔍 TRAINING VALIDATION REPORT")
        logger.info("=" * 50)
        
        # 1. Dataset Size Validation
        train_size = train_dataset.num_rows if hasattr(train_dataset, 'num_rows') else 0
        valid_size = valid_dataset.num_rows if valid_dataset and hasattr(valid_dataset, 'num_rows') else 0
        test_size = test_dataset.num_rows if test_dataset and hasattr(test_dataset, 'num_rows') else 0
        
        logger.info(f"📊 Dataset Sizes:")
        logger.info(f"   Train: {train_size:,} samples")
        logger.info(f"   Valid: {valid_size:,} samples") 
        logger.info(f"   Test:  {test_size:,} samples")
        
        # Check for minimum viable sizes
        if train_size < 1000:
            logger.error("❌ CRITICAL: Training set too small (< 1,000 samples)")
        elif train_size < 10000:
            logger.warning("⚠️  WARNING: Training set quite small (< 10,000 samples)")
        else:
            logger.info("✅ Training set size looks adequate")
            
        if valid_size < 100:
            logger.error("❌ CRITICAL: Validation set too small (< 100 samples)")
        elif valid_size < 1000:
            logger.warning("⚠️  WARNING: Validation set quite small (< 1,000 samples)")
        else:
            logger.info("✅ Validation set size looks adequate")
        
        # 2. Check for Data Leakage
        leakage_ok = self._check_data_leakage(train_dataset, valid_dataset, test_dataset)
        
        # 3. Validate ID Mappings
        self._validate_id_mappings()
        
        # 4. Check Embedding Quality
        self._validate_embeddings()
        
        # 5. Analyze Training Metrics
        self._analyze_training_metrics()
        
        logger.info("=" * 50)
        
        if not leakage_ok:
            logger.error("❌ VALIDATION FAILED: Data leakage detected")
            return False
        else:
            logger.info("✅ VALIDATION PASSED: No critical issues detected")
            return True
    def train_model(self, datasets, epochs=15, batch_size=64, wandb_entity=None, wandb_project="smokeshow_2tower_model", 
                wandb_run_name=None, learning_rate=0.001, num_users=15, max_interactions_per_user=7, 
                num_cold_users=10, num_cold_items=10, processed_data_dir=None, force_reprocess=False):
        """
        Train the recommendation model with better handling of small datasets.
        """
        # Initialize logging and extract interaction data
        logger.info("Starting end-to-end training of recommendation models")
        interaction_dict = datasets["interaction_dict"]
        
        # Initialize W&B logging
        run_config = self._setup_wandb(wandb_entity, wandb_project, wandb_run_name, 
                                    epochs, batch_size, learning_rate, num_users, 
                                    max_interactions_per_user, num_cold_users, num_cold_items)
        
        # Analyze user data and create global ID mappings
        user_id_map, video_id_map = self._create_global_id_mappings(interaction_dict)
        
        # Process interactions for each data split with proper user-level splitting
           # Process interactions or load pre-processed data
        train_dir, valid_dir, test_dir, user_id_map, video_id_map = self.load_or_process_data(
            interaction_dict, batch_size, user_id_map, video_id_map, 
            force_reprocess=force_reprocess, processed_data_dir=processed_data_dir
        )
        
        # Update schema with mapping cardinalities
        self._update_schema_cardinalities(user_id_map, video_id_map)
        
        # Create datasets with proper batch sizing
        train_dataset, valid_dataset, test_dataset = self._create_datasets(
            train_dir, valid_dir, test_dir, batch_size)
        self.validate_training_results(train_dataset, valid_dataset, test_dataset)
        # Build model if needed
        if self.model is None:
            self._build_model()
        
        # Train the model with improved error handling
        self._train_with_callbacks(train_dataset, valid_dataset, epochs, batch_size)
        
        # Evaluate on test set if available
        if test_dataset is not None:
            self._evaluate_model(test_dataset)
            
        else:
            logger.warning("Skipping test evaluation due to insufficient test data")
        
        # Save model and finish
        self.save_model()
        logger.info("End-to-end training and evaluation complete")
        wandb.finish()
    
    def _setup_wandb(self, wandb_entity, wandb_project, wandb_run_name, 
                    epochs, batch_size, learning_rate, num_users, 
                    max_interactions_per_user, num_cold_users, num_cold_items):
        """Set up Weights & Biases tracking."""
        os.environ["WANDB_API_KEY"] = "512fcfc1ad2709362c11ff6fe2471e662e1f5edd"
        
        if wandb_run_name is None:
            wandb_run_name = (
                f"two_tower_bs{batch_size}_lr{learning_rate}_ep{epochs}_"
                f"nu{num_users}_mi{max_interactions_per_user}_cu{num_cold_users}_ci{num_cold_items}_dr_{self.user_embedding_model.dropout}_l2_{self.user_embedding_model.l2_reg}_id_"
                f"{int(time.time())}"
            )
            
        config = {
            "embedding_dim": self.embedding_dim,
            "max_sequence_length": self.user_embedding_model.max_sequence_length,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_users": num_users,
            "max_interactions_per_user": max_interactions_per_user,
            "num_cold_users": num_cold_users,
            "num_cold_items": num_cold_items,
            "dropout": self.user_embedding_model.dropout,
            "l2_reg": self.user_embedding_model.l2_reg
        }
        
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config=config
        )
        
        return config
    
    def _create_global_id_mappings(self, interaction_dict):
        """Create global ID mappings for users and videos."""
        # Analyze user types
        real_users = [uid for uid in interaction_dict.keys() if not uid.startswith("cold_user_")]
        cold_users = [uid for uid in interaction_dict.keys() if uid.startswith("cold_user_")]
        logger.info(f"Interaction dict: {len(real_users)} real users, {len(cold_users)} cold-start users, "
                    f"total {len(interaction_dict)} users")
        
        # Create user ID mapping
        user_ids = sorted(set(interaction_dict.keys()))
        user_id_map = {user_id: idx + 1 for idx, user_id in enumerate(user_ids)}
        
        # Debug: Log sample of user ID mapping
        logger.debug(f"User ID mapping sample (first 10): {dict(list(user_id_map.items())[:10])}")
        logger.debug(f"User ID mapping sample (last 10): {dict(list(user_id_map.items())[-10:])}")
        
        # Create video ID mapping
        video_ids = set()
        for interactions in interaction_dict.values():
            for inter in interactions:
                video_id = str(inter.get('video_id'))
                if video_id:
                    video_ids.add(video_id)
        
        video_id_map = {video_id: idx + 1 for idx, video_id in enumerate(sorted(video_ids))}
        
        # Debug: Log sample of video ID mapping
        logger.debug(f"Video ID mapping sample (first 10): {dict(list(video_id_map.items())[:10])}")
        logger.debug(f"Video ID mapping sample (last 10): {dict(list(video_id_map.items())[-10:])}")
        
        logger.info(f"Global ID mappings created: {len(user_id_map)} users, {len(video_id_map)} videos")
        
        # Debug: Save mappings to JSON for inspection
        with open("user_id_mapping_debug.json", "w") as f:
            json.dump(user_id_map, f, indent=2)
        with open("video_id_mapping_debug.json", "w") as f:
            json.dump(video_id_map, f, indent=2)
        logger.debug("Saved ID mappings to JSON files for debugging")
        
        return user_id_map, video_id_map
    
    def _update_schema_cardinalities(self, user_id_map, video_id_map):
        """Update schema with the correct cardinalities based on ID mappings."""
        if self.schema is None:
            self._create_schema()
        
        self.schema.column_schemas['user_id'].properties['domain']['max'] = len(user_id_map)
        self.schema.column_schemas['video_id'].properties['domain']['max'] = len(video_id_map)
        
        # Store mappings for future use
        self.user_id_map = user_id_map
        self.video_id_map = video_id_map
        
        logger.info(f"Updated schema: user_id max={len(user_id_map)}, video_id max={len(video_id_map)}")
    
    def _create_datasets(self, train_dir, valid_dir, test_dir, batch_size):
        """
        Create datasets from processed files and verify them with proper batch sizing.
        """
        # Find all parquet files
        train_files = glob.glob(os.path.join(train_dir, "*.parquet"))
        valid_files = glob.glob(os.path.join(valid_dir, "*.parquet"))
        test_files = glob.glob(os.path.join(test_dir, "*.parquet"))
        
        # Check the size of each dataset
        train_size = sum(pd.read_parquet(f).shape[0] for f in train_files) if train_files else 0
        valid_size = sum(pd.read_parquet(f).shape[0] for f in valid_files) if valid_files else 0
        test_size = sum(pd.read_parquet(f).shape[0] for f in test_files) if test_files else 0
        
        logger.info(f"Dataset sizes - Train: {train_size}, Valid: {valid_size}, Test: {test_size}")
        
        # Adjust batch sizes based on dataset sizes to prevent issues
        train_batch_size = min(batch_size, train_size) if train_size > 0 else batch_size
        valid_batch_size = min(batch_size, valid_size) if valid_size > 0 else batch_size
        test_batch_size = min(batch_size, test_size) if test_size > 0 else batch_size
        
        # Ensure minimum batch size for InBatchSampler
        min_batch_size = 16  # Minimum size for effective sampling
        train_batch_size = max(min_batch_size, train_batch_size)
        valid_batch_size = max(min_batch_size, valid_batch_size) if valid_size > min_batch_size else valid_size
        test_batch_size = max(min_batch_size, test_batch_size) if test_size > min_batch_size else test_size
        
        logger.info(f"Adjusted batch sizes - Train: {train_batch_size}, Valid: {valid_batch_size}, Test: {test_batch_size}")
        
        # Create datasets with adjusted batch sizes
        train_dataset = Dataset(train_files, schema=self.schema, batch_size=train_batch_size)
        valid_dataset = Dataset(valid_files, schema=self.schema, batch_size=valid_batch_size)
        test_dataset = Dataset(test_files, schema=self.schema, batch_size=test_batch_size)
        
    
        return train_dataset, valid_dataset, test_dataset
    
    def _train_with_callbacks(self, train_dataset, valid_dataset, epochs, batch_size):
        """
        Train the model with appropriate callbacks, handling small validation sets.
        """
        logger.info("Starting model training...")
        
        # Create callbacks
        callbacks = []
        
        if valid_dataset is not None:
            wandb_callback = WandbCallback(validation_data=valid_dataset, batch_size=batch_size)
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_ndcg_at_10',
                patience=5,
                mode='max',
                restore_best_weights=True
            )
            lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_ndcg_at_10',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                mode='max'
            )
            callbacks = [wandb_callback, early_stopping, lr_scheduler]
        else:
            # If no validation data, monitor training metrics
            wandb_callback = WandbCallback(validation_data=None, batch_size=batch_size)
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=5,
                mode='min',
                restore_best_weights=True
            )
            callbacks = [wandb_callback]
        
        # Train the model
        try:
            self.model.fit(
                train_dataset,
                validation_data=valid_dataset,
                epochs=epochs,
                callbacks=callbacks,
                batch_size=batch_size,
            )
        except tf.errors.InvalidArgumentError as e:
            logger.error(f"Training failed with error: {str(e)}")
            logger.error("Attempting to continue training without validation...")
            if valid_dataset is not None:
                # Retry without validation
                self.model.fit(
                    train_dataset,
                    epochs=epochs,
                    callbacks=[wandb_callback]
                )
        
        logger.info("Model training complete")

   

    def _create_evaluation_schema_and_pool(self):
        """
        Create a dedicated evaluation schema and candidate pool with all item IDs.
        This function runs once and creates resources needed for all evaluations.
        """
        logger.info("Creating unified evaluation schema and candidate pool...")
        
        # 1. Get datasets from all splits
        datasets = {}
        for split_name in ['train', 'valid', 'test']:
            split_path = f"processed_data_{split_name}"
            if os.path.exists(split_path):
                files = glob.glob(os.path.join(split_path, "*.parquet"))
                if files:
                    datasets[split_name] = Dataset(files, schema=self.schema)
                    logger.info(f"Loaded {split_name} dataset with {datasets[split_name].num_rows} rows")
        
        if not datasets:
            logger.error("No datasets found for evaluation")
            return None, None
        
        # 2. Extract item features and combine using pandas/dask instead of concatenate
        all_candidate_dfs = []
        
        for split_name, dataset in datasets.items():
            # Extract unique items from this split
            split_candidates = unique_rows_by_features(dataset, Tags.ITEM, Tags.ITEM_ID)
            logger.info(f"Found {split_candidates.num_rows} unique items in {split_name}")
            
            # Convert to DataFrame for easier merging
            if hasattr(split_candidates, 'to_ddf'):
                # Convert to dask dataframe then compute to pandas
                split_df = split_candidates.to_ddf().compute()
                
                # If result is a cuDF DataFrame, convert to pandas
                if hasattr(split_df, 'to_pandas'):
                    split_df = split_df.to_pandas()
            else:
                # Already a pandas DataFrame or similar
                split_df = split_candidates.compute()
            
            all_candidate_dfs.append(split_df)
        
        # Combine all DataFrames
        if not all_candidate_dfs:
            logger.error("No candidate DataFrames found")
            return None, None
        
        # Concatenate DataFrames
        combined_df = pd.concat(all_candidate_dfs, ignore_index=True)
        
        # Drop duplicates by item ID
        item_id_col = 'video_id'  # Adjust if your item ID column has a different name
        combined_df = combined_df.drop_duplicates(subset=[item_id_col])
        
        logger.info(f"Combined candidate pool has {len(combined_df)} unique items")
        
        # Create Dataset from combined DataFrame
        item_schema = self.schema.select_by_tag(Tags.ITEM)
        all_candidates = Dataset(combined_df, schema=item_schema)
        
        # 3. Create evaluation schema with TARGET tag
        target_schema = self._ensure_target_tagging(self.schema)
        
        return all_candidates, target_schema

    def _ensure_target_tagging(self, schema):
        """
        Ensure the item ID column has the TARGET tag.
        """
        # Create a modified schema with TARGET tag on item ID
        new_columns = []
        for col_name, col_schema in schema.column_schemas.items():
            if col_name == 'video_id':  # Adjust if your item ID has a different name
                # Add TARGET tag
                tags = list(col_schema.tags)
                if Tags.TARGET not in tags:
                    tags.append(Tags.TARGET)
                
                new_col = ColumnSchema(
                    name=col_name,
                    tags=tags,
                    dtype=col_schema.dtype,
                    properties=col_schema.properties
                )
                new_columns.append(new_col)
            else:
                new_columns.append(col_schema)
        
        target_schema = Schema(new_columns)
        
        # Verify target column is properly tagged
        target_cols = target_schema.select_by_tag(Tags.TARGET).column_names
        logger.info(f"Target columns in evaluation schema: {target_cols}")
        
        return target_schema
    def _evaluate_model(self, test_dataset):
        """
        Evaluate the model using a clean approach with pre-created resources.
        """
        if test_dataset is None or test_dataset.num_rows == 0:
            logger.warning("No test dataset available, skipping evaluation")
            return
        
        # Basic evaluation
        logger.info("Running basic evaluation...")
        try:
            test_results = self.model.evaluate(test_dataset, batch_size=64, return_dict=True)
            logger.info(f"Test results: {test_results}")
            wandb.log({f"test_{k}": v for k, v in test_results.items()})
        except Exception as e:
            logger.error(f"Error in basic evaluation: {str(e)}")
        
        # Create or get evaluation resources (only compute once)
        if not hasattr(self, '_eval_candidate_pool') or self._eval_candidate_pool is None:
            self._eval_candidate_pool, self._eval_target_schema = self._create_evaluation_schema_and_pool()
            
        if self._eval_candidate_pool is None:
            logger.error("Failed to create evaluation resources")
            return
            
        # Apply target schema to test dataset
        test_with_target = Dataset(test_dataset.to_ddf(), schema=self._eval_target_schema)
        
        # Create top-k encoder
        logger.info("Creating top-k encoder for evaluation...")
        topk = 10
        topk_model = self.model.to_top_k_encoder(
            self._eval_candidate_pool,
            k=topk,
            batch_size=32
        )
        
        # Compile and evaluate
        topk_model.compile(run_eagerly=False)
        eval_loader = mm.Loader(test_with_target, batch_size=32)
        
        logger.info("Running top-k evaluation...")
        metrics = topk_model.evaluate(eval_loader, return_dict=True)
        logger.info(f"Top-{topk} evaluation metrics: {metrics}")
        
        # Log metrics
        wandb.log({f"topk_{k}": v for k, v in metrics.items()})
        
        return metrics
class VideoFeatureManager:
    def __init__(self, faiss_api_url= None, api_key=None):
        self.faiss_api_url = faiss_api_url
        self.api_key = api_key
        self.embedding_cache = {}
    
    def get_video_features(self, video_id, include_metadata=True):
        embedding = self.get_video_embedding(video_id)
        
        if include_metadata:
            metadata = self.get_video_metadata(video_id)
            metadata_vector = create_video_meta_vector(metadata)
            return {'video_id': video_id, 'embedding': embedding, 'metadata': metadata, 'metadata_vector': metadata_vector}
        else:
            return {'video_id': video_id, 'embedding': embedding}
    
    def get_video_embedding(self, video_id):
        video_id_str = str(video_id)
        if video_id_str in self.embedding_cache:
            return self.embedding_cache[video_id_str]
        
        # Get internvideo embedding from S3
        embedding = get_internvideo_embedding_from_s3(video_id)
        
        if embedding is not None:
            self.embedding_cache[video_id_str] = embedding
            return embedding
        
        # Use mean embedding instead of random
        if not hasattr(self, 'mean_embedding') or self.mean_embedding is None:
            self._compute_mean_embedding()
        
        logger.warning(f"Embedding not found for video {video_id}, using mean embedding")
        self.embedding_cache[video_id_str] = self.mean_embedding
        return self.mean_embedding

    def _compute_mean_embedding(self):
        """Compute the mean embedding vector across all available videos."""
        # Start with a smaller sample to avoid memory issues
        sample_size = 500
        embeddings = []
        count = 0
        
        # Collect a sample of available embeddings
        for video_id, embedding in self.embedding_cache.items():
            if embedding is not None and not np.all(embedding == 0):
                embeddings.append(embedding)
                count += 1
                if count >= sample_size:
                    break
        
        # If we have no embeddings yet, fall back to zeros
        if not embeddings:
            self.mean_embedding = np.zeros(256, dtype=np.float32)
        else:
            # Compute the mean
            self.mean_embedding = np.mean(embeddings, axis=0, dtype=np.float32)
        
        logger.info(f"Computed mean embedding from {len(embeddings)} videos")
    
    def get_video_metadata(self, video_id):
        DB_CONFIG = {
            "host": "database-2-new.cz88k0kmy8tp.us-west-2.rds.amazonaws.com",
            "port": "5432",
            "database": "smoke",
            "user": "postgres",
            "password": "LKX596BysURhjmTd8kgVpn"
        }
        
        try:
            conn = pg8000.connect(**DB_CONFIG)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT vm.*, c.category_name_en, c.category_name_cn 
                FROM video_meta vm
                LEFT JOIN categories c ON vm.category_id = c.category_id
                WHERE vm.item_id = %s
            """, (video_id,))
            columns = [desc[0] for desc in cursor.description]
            result = cursor.fetchone()
            if result:
                metadata = dict(zip(columns, result))
                cursor.close()
                conn.close()
                return metadata
            cursor.close()
            conn.close()
            return {}
        except Exception as e:
            logger.error(f"Error fetching metadata for video {video_id}: {str(e)}")
            return {}

DB_CONFIG = {
    "host": "database-2-new.cz88k0kmy8tp.us-west-2.rds.amazonaws.com",
    "port": "5432",
    "database": "smoke",
    "user": "postgres",
    "password": "LKX596BysURhjmTd8kgVpn"
}

text_embedding_cache = {}

def get_db_connection():
    return pg8000.connect(**DB_CONFIG)

def get_category_info(cursor, category_id, level='category_id'):
    if not category_id:
        return None
    try:
        cursor.execute(f"""
            SELECT category_id, category_name_cn, category_name_en, category_level, parent_id, root_id
            FROM categories 
            WHERE {level} = %s
        """, (str(category_id),))
        result = cursor.fetchone()
        if result:
            return {'id': result[0], 'name_cn': result[1], 'name_en': result[2], 'level': result[3], 'parent_id': result[4], 'root_id': result[5]}
        return None
    except Exception as e:
        logger.warning(f"Error looking up category {category_id} ({level}): {str(e)}")
        return None

def get_internvideo_embedding_from_s3(video_id):
    """Fetch internvideo embedding from S3, with caching."""
    video_id_str = str(video_id)
    cache_key = f"internvideo_{video_id_str}"
    
    if cache_key in embedding_cache:
        return embedding_cache[cache_key]
    
    internvideo_prefix = "internvideo_embedding/"
    s3_key = f"{internvideo_prefix}{video_id_str}.npy"
    
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        embedding_bytes = response['Body'].read()
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        
        # Ensure the embedding is 256-dimensional
        if embedding.shape[0] != 256:
            logger.warning(f"Internvideo embedding for video {video_id_str} has shape {embedding.shape}, expected (256,). Reshaping.")
            if embedding.shape[0] > 256:
                embedding = embedding[:256]  # Truncate if larger
            else:
                # Pad with zeros if smaller
                padding = np.zeros(256 - embedding.shape[0], dtype=np.float32)
                embedding = np.concatenate([embedding, padding])
                
        embedding_cache[cache_key] = embedding
        logger.debug(f"Successfully fetched internvideo embedding for video {video_id_str}")
        # print("found  internvideo embedding sucessfully {video_id_str} at {s3_key}")
        return embedding
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.warning(f"No internvideo embedding found in S3 for video {video_id_str} at {s3_key}")
        else:
            logger.error(f"S3 error fetching internvideo embedding for video {video_id_str}: {str(e)}")
        embedding_cache[cache_key] = None
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching internvideo embedding for video {video_id_str}: {str(e)}")
        embedding_cache[cache_key] = None
        return None
    
def get_embedding_from_s3(video_id):
    """Fetch precomputed embedding from S3, with caching."""
    video_id_str = str(video_id)
    if video_id_str in embedding_cache:
        return embedding_cache[video_id_str]
    
    s3_key = f"{S3_PREFIX}{video_id_str}.npy"
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        embedding_bytes = response['Body'].read()
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        embedding_cache[video_id_str] = embedding
        logger.debug(f"Successfully fetched embedding for video {video_id_str}")
        return embedding
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.warning(f"No embedding found in S3 for video {video_id_str} at {s3_key}")
        else:
            logger.error(f"S3 error fetching embedding for video {video_id_str}: {str(e)}")
        embedding_cache[video_id_str] = None
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching embedding for video {video_id_str}: {str(e)}")
        embedding_cache[video_id_str] = None
        return None

def create_video_meta_vector(video_meta, cursor=None, use_openai=False):
    """Create feature vector using precomputed embedding from S3."""
    vector = []
    
    if 'duration' in video_meta:
        try:
            duration = float(video_meta['duration']) if isinstance(video_meta['duration'], str) else video_meta['duration']
            norm_duration = min(duration / 300, 1.0)
            vector.append(norm_duration)
        except (ValueError, TypeError):
            vector.append(0)
    else:
        vector.append(0)
    
    if 'author_fans_count' in video_meta:
        try:
            fans_count = float(video_meta['author_fans_count']) if isinstance(video_meta['author_fans_count'], str) else video_meta['author_fans_count']
            norm_fans = min(fans_count / 1000000, 1.0)
            vector.append(norm_fans)
        except (ValueError, TypeError):
            vector.append(0)
    else:
        vector.append(0)
    
    video_id = video_meta.get('video_id', '')
    embedding = get_embedding_from_s3(video_id) if video_id else None
    
    if embedding is None:
        embedding = np.zeros(1536)
    
    vector = np.concatenate([np.array(vector), embedding])
    
    return np.array(vector, dtype=np.float32)

def create_interaction_vector(interaction, video_meta):
    vector = []
    
    if 'exposed_time' in interaction:
        try:
            exposed_time = int(interaction['exposed_time']) if isinstance(interaction['exposed_time'], str) else interaction['exposed_time']
            timestamp = datetime.fromtimestamp(exposed_time)
            hour = timestamp.hour
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_of_week = timestamp.weekday()
            dow_sin = np.sin(2 * np.pi * day_of_week / 7)
            dow_cos = np.cos(2 * np.pi * day_of_week / 7)
            vector.extend([hour_sin, hour_cos, dow_sin, dow_cos])
        except (ValueError, TypeError):
            vector.extend([0, 0, 0, 0])
    elif 'p_hour' in interaction:
        try:
            hour = int(interaction['p_hour']) if isinstance(interaction['p_hour'], str) else interaction['p_hour']
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            vector.extend([hour_sin, hour_cos, 0, 0])
        except (ValueError, TypeError):
            vector.extend([0, 0, 0, 0])
    else:
        vector.extend([0, 0, 0, 0])
    
    if 'watch_time' in interaction and 'duration' in video_meta:
        try:
            watch_time = float(interaction['watch_time']) if isinstance(interaction['watch_time'], str) else interaction['watch_time']
            duration = float(video_meta['duration']) if isinstance(video_meta['duration'], str) else video_meta['duration']
            completion_rate = min(watch_time / max(duration, 0.1), 1.0)
            vector.append(completion_rate)
        except (ValueError, TypeError, ZeroDivisionError):
            vector.append(0)
    elif 'watch_time' in interaction:
        try:
            watch_time = float(interaction['watch_time']) if isinstance(interaction['watch_time'], str) else interaction['watch_time']
            norm_watch_time = min(watch_time / 300, 1.0)
            vector.append(norm_watch_time)
        except (ValueError, TypeError):
            vector.append(0)
    else:
        vector.append(0)
    
    bool_features = ['cvm_like', 'comment', 'follow', 'collect', 'forward', 'effective_view', 'hate']
    for feature in bool_features:
        if feature in interaction:
            try:
                value = interaction[feature]
                bool_value = value.lower() in ('true', 't', 'yes', 'y', '1') if isinstance(value, str) else bool(value)
                vector.append(1 if bool_value else 0)
            except Exception:
                vector.append(0)
        else:
            vector.append(0)
    
    return np.array(vector, dtype=np.float32)

def get_s3_video_ids():
    """Fetch list of video IDs from S3 embeddings."""
    s3 = boto3.client('s3')
    video_ids = []
    continuation_token = None
    
    logger.info(f"Listing video IDs from s3://{S3_BUCKET}/{S3_PREFIX}")
    try:
        while True:
            list_kwargs = {
                'Bucket': S3_BUCKET,
                'Prefix': S3_PREFIX,
                'MaxKeys': 1000
            }
            if continuation_token:
                list_kwargs['ContinuationToken'] = continuation_token
            
            response = s3.list_objects_v2(**list_kwargs)
            for obj in response.get('Contents', []):
                key = obj['Key']
                if key.endswith('.npy'):
                    video_id = key[len(S3_PREFIX):-4]
                    video_ids.append(video_id)
            
            continuation_token = response.get('NextContinuationToken')
            if not continuation_token:
                break
        
        logger.info(f"Found {len(video_ids)} video IDs in S3")
        return video_ids
    except Exception as e:
        logger.error(f"Failed to list S3 video IDs: {str(e)}")
        return []

def fetch_user_ids(conn, num_users):
    """
    Fetch the first num_users user IDs from the database in ascending order.
    """
    cursor = conn.cursor()
    try:
        start_time = time.time()
        cursor.execute("""
            SELECT DISTINCT user_id
            FROM interaction_filtered
            ORDER BY user_id
            LIMIT %s
        """, (num_users,))
        user_ids = [row[0] for row in cursor.fetchall()]
        logger.info(f"Fetched {len(user_ids)} user IDs in {time.time() - start_time:.2f}s")
        return user_ids
    except pg8000.exceptions.DatabaseError as e:
        logger.error(f"Failed to fetch user IDs: {str(e)}")
        return []
    finally:
        cursor.close()

def save_user_ids(user_ids, filename="user_ids.json"):
    """
    Save the list of user IDs to a JSON file.
    """
    try:
        with open(filename, "w") as f:
            json.dump(user_ids, f)
        logger.info(f"Saved {len(user_ids)} user IDs to {filename}")
    except Exception as e:
        logger.error(f"Failed to save user IDs to {filename}: {str(e)}")

def fetch_interactions(conn, user_ids, batch_size):
    """
    Fetch interactions for a batch of users from the database.
    """
    cursor = conn.cursor()
    all_interactions = {}
    try:
        for i in range(0, len(user_ids), batch_size):
            batch_users = user_ids[i:i + batch_size]
            placeholders = ','.join(['%s'] * len(batch_users))
            cursor.execute(f"""
                SELECT i.*, vm.*
                FROM interaction_filtered i
                LEFT JOIN video_metadata vm ON i.pid = vm.video_id
                WHERE i.user_id IN ({placeholders})
                ORDER BY i.user_id, i.exposed_time
            """, tuple(batch_users))
            
            raw_interactions = [dict(zip([desc[0] for desc in cursor.description], row)) 
                               for row in cursor.fetchall()]
            for inter in raw_interactions:
                user_id = inter['user_id']
                if user_id not in all_interactions:
                    all_interactions[user_id] = []
                all_interactions[user_id].append(inter)
            
            logger.info(f"Processed interaction batch {i//batch_size + 1} for {len(batch_users)} users")
    except Exception as e:
        logger.error(f"Failed to fetch interactions: {str(e)}")
    finally:
        cursor.close()
    return all_interactions

def process_user_interactions(user_id, interactions, max_interactions_per_user, cursor, video_metadata_cache, use_openai_embeddings):
    """
    Process and validate interactions for a single user.
    """
    valid_interactions = []
    for inter in interactions[:max_interactions_per_user]:
        item_id = inter.get('pid')
        if not item_id:
            continue
            
        if item_id in video_metadata_cache:
            video_meta = video_metadata_cache[item_id]
        else:
            video_meta = {
                'video_id': item_id,
                'title': inter.get('title', ''),
                'asr_text': '',
                'duration': '300'
            }
            video_metadata_cache[item_id] = video_meta
            
        video_id = video_meta.get('video_id', item_id)
        if get_embedding_from_s3(video_id) is None:
            continue
            
        if 'video_meta_vector' not in inter:
            inter['video_meta_vector'] = create_video_meta_vector(video_meta, cursor=cursor, use_openai=use_openai_embeddings)
        if inter['video_meta_vector'] is None:
            inter['video_meta_vector'] = np.zeros(1538, dtype=np.float32)
            
        if 'interaction_vector' not in inter:
            inter['interaction_vector'] = create_interaction_vector(inter, video_meta)
        if inter['interaction_vector'] is None:
            inter['interaction_vector'] = np.zeros(12, dtype=np.float32)
            
        inter.update({
            'video_id': inter['pid'],
            'watch_time': float(inter.get('watch_time', '0') or 0),
            'duration': float(video_meta.get('duration', '300') or 300),
            'liked': inter.get('cvm_like', False),
            'shared': inter.get('forward', False),
            'author_id': video_meta.get('author_id', ''),
            'follower_count': float(video_meta.get('follower_count', '0') or 0),
            'following_count': float(video_meta.get('following_count', '0') or 0),
            'video_meta': video_meta
        })
        
        valid_interactions.append(inter)
    
    return valid_interactions

def generate_cold_start_data(
    num_cold_users: int,
    num_cold_items: int,
    interaction_dict: Dict,
    video_metadata_cache: Dict,
    user_id_map: Dict,
    video_id_map: Dict,
    random_seed: int = 42
) -> tuple[Dict, Dict, Dict]:
    """
    Generate cold-start users with exactly 3 interactions using popular videos, and cold-start items for interactions only.
    Updates interaction_dict, video_metadata_cache, user_id_map, and video_id_map.
    
    Args:
        num_cold_users: Number of cold-start users to generate.
        num_cold_items: Number of cold-start items to generate.
        interaction_dict: Dictionary of user interactions.
        video_metadata_cache: Cache of video metadata.
        user_id_map: Mapping of user IDs to integers.
        video_id_map: Mapping of video IDs to integers.
        random_seed: Random seed for reproducibility.
    
    Returns:
        Tuple of updated (interaction_dict, user_id_map, video_id_map).
    """
    if num_cold_users == 0 and num_cold_items == 0:
        logger.info("No cold-start data requested, skipping generation")
        return interaction_dict, user_id_map, video_id_map
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    logger.info(f"Generating {num_cold_users} cold-start users and {num_cold_items} cold-start items")
    
    # Fetch available video IDs from S3
    s3_video_ids = get_s3_video_ids()
    if not s3_video_ids:
        logger.error("No video IDs found in S3, cannot generate cold-start data")
        return interaction_dict, user_id_map, video_id_map
    
    # Identify interacted videos
    interacted_videos = set()
    for interactions in interaction_dict.values():
        for inter in interactions:
            if inter.get('video_id'):
                interacted_videos.add(str(inter['video_id']))
    
    # Select cold-start items (non-interacted videos)
    available_cold_items = [vid for vid in s3_video_ids if vid not in interacted_videos]
    if len(available_cold_items) < num_cold_items:
        logger.warning(f"Only {len(available_cold_items)} cold-start items available, expected {num_cold_items}")
        num_cold_items = len(available_cold_items)
    
    cold_items = random.sample(available_cold_items, min(num_cold_items, len(available_cold_items)))
    
    # Add cold-start items to video_metadata_cache and video_id_map
    for video_id in cold_items:
        if video_id not in video_metadata_cache:
            video_metadata_cache[video_id] = {
                'video_id': video_id,
                'duration': str(random.randint(30, 600)),  # Random duration between 30s and 10min
                'title': f"Synthetic Video {video_id}",
                'asr_text': '',
                'author_fans_count': str(random.randint(0, 10000))
            }
            logger.debug(f"Created cold-start item {video_id} with metadata: {video_metadata_cache[video_id]}")
        
        # Update video_id_map if not already present
        if str(video_id) not in video_id_map:
            video_id_map[str(video_id)] = len(video_id_map) + 1
            logger.debug(f"Added cold-start video {video_id} to video_id_map with ID {video_id_map[str(video_id)]}")
    
    # Generate cold-start users
    existing_user_ids = set(interaction_dict.keys())
    cold_user_interactions = {}
    
    # Select popular videos for cold-start user interactions (exclude cold items)
    popular_videos = s3_video_ids[:100]  # Top 100 videos, assumed to be popular
    if not popular_videos:
        logger.error("No popular videos available for cold-start user interactions")
        return interaction_dict, user_id_map, video_id_map
    
    for i in range(num_cold_users):
        cold_user_id = f"cold_user_{i}"
        while cold_user_id in existing_user_ids or cold_user_id in cold_user_interactions:
            i += 1
            cold_user_id = f"cold_user_{i}"
        
        # Add to user_id_map
        if str(cold_user_id) not in user_id_map:
            user_id_map[str(cold_user_id)] = len(user_id_map) + 1
            logger.debug(f"Added cold-start user {cold_user_id} to user_id_map with ID {user_id_map[str(cold_user_id)]}")
        
        # Generate exactly 3 interactions
        num_interactions = 3
        interactions = []
        current_time = int(time.time())
        
        for j in range(num_interactions):
            # Select a popular video (non-cold item)
            video_id = random.choice(popular_videos)
            video_meta = video_metadata_cache.get(video_id, {
                'video_id': video_id,
                'duration': '300',
                'title': '',
                'asr_text': '',
                'author_fans_count': '0'
            })
            
            # Generate interaction
            interaction = {
                'user_id': cold_user_id,
                'video_id': video_id,
                'pid': video_id,  # Match original script
                'exposed_time': current_time - random.randint(0, 86400),  # Within last 24 hours
                'cvm_like': random.choice([True, False]),
                'comment': False,
                'collect': random.choice([True, False]),
                'video_meta': video_meta
            }
            
            # Add vectors
            interaction['video_meta_vector'] = create_video_meta_vector(video_meta)
            interaction['interaction_vector'] = create_interaction_vector(interaction, video_meta)
            
            interactions.append(interaction)
            logger.debug(f"Generated interaction {j+1} for cold-start user {cold_user_id}: "
                        f"video_id={video_id}", 
                        f"like={interaction['cvm_like']}")
        
        cold_user_interactions[cold_user_id] = interactions
        logger.info(f"Created cold-start user {cold_user_id} with {len(interactions)} interactions")
    
    # Merge cold-start users into interaction_dict
    interaction_dict.update(cold_user_interactions)
    
    # Validate cold-start data
    actual_cold_users = len([uid for uid in interaction_dict if uid.startswith("cold_user_")])
    actual_cold_items = len(cold_items)
    logger.info(f"Validation: Created {actual_cold_users} cold-start users (expected {num_cold_users}), "
                f"{actual_cold_items} cold-start items (expected {num_cold_items})")
    
    if actual_cold_users < num_cold_users:
        logger.warning(f"Created fewer cold-start users than expected: {actual_cold_users}/{num_cold_users}")
    
    # Verify interactions
    for cold_user_id, interactions in cold_user_interactions.items():
        if len(interactions) != 3:
            logger.error(f"Cold-start user {cold_user_id} has {len(interactions)} interactions, expected exactly 3")
        for i, inter in enumerate(interactions):
            if not inter.get('video_id') or not inter.get('video_meta_vector').shape == (1538,) or \
               not inter.get('interaction_vector').shape == (12,):
                logger.error(f"Invalid interaction {i} for cold-start user {cold_user_id}: {inter}")
            if inter['video_id'] in cold_items:
                logger.error(f"Cold-start user {cold_user_id} interaction {i} uses cold item {inter['video_id']}, expected popular video")
    
    # Save sample of cold-start data for debugging
    sample_cold_users = dict(list(cold_user_interactions.items())[:5])
    serializable_sample = {}
    for uid, inters in sample_cold_users.items():
        serializable_inters = []
        for inter in inters:
            # Convert NumPy arrays to lists for JSON serialization
            serializable_inter = inter.copy()
            serializable_inter['video_meta_vector'] = inter['video_meta_vector'].tolist()
            serializable_inter['interaction_vector'] = inter['interaction_vector'].tolist()
            # Convert any other NumPy types (e.g., np.bool_)
            for key, value in serializable_inter.items():
                if isinstance(value, np.ndarray):
                    serializable_inter[key] = value.tolist()
                elif isinstance(value, np.bool_):
                    serializable_inter[key] = bool(value)
                elif isinstance(value, np.integer):
                    serializable_inter[key] = int(value)
                elif isinstance(value, np.floating):
                    serializable_inter[key] = float(value)
            serializable_inters.append(serializable_inter)
        serializable_sample[uid] = serializable_inters
    
    with open("cold_start_sample.json", "w") as f:
        json.dump(serializable_sample, f, indent=2)
    logger.info("Saved sample of cold-start data to cold_start_sample.json")
    
    return interaction_dict, user_id_map, video_id_map

def prepare_training_data(use_openai_embeddings=False, num_users=10, max_interactions_per_user=15, 
                         num_cold_users=50, num_cold_items=30, batch_size=64, random_seed=42):
    """
    Prepare training data with memory-efficient chunk processing, including cold-start generation.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    logger.info("Connecting to database")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    interaction_dict = {}
    video_metadata_cache = {}
    user_id_map = {}  # Initialize ID mappings
    video_id_map = {}
    
    # Fetch more users than needed to ensure we get exactly num_users after filtering
    fetch_multiplier = 1.0
    fetch_num_users = int(num_users * fetch_multiplier)
    
    user_ids = fetch_user_ids(conn, fetch_num_users)
    if not user_ids:
        logger.error("No user IDs fetched, aborting data preparation")
        cursor.close()
        conn.close()
        return {"interaction_dict": {}, "num_users": 0, "user_id_map": {}, "video_id_map": {}}
    
    logger.info(f"Successfully fetched {len(user_ids)} user IDs")
    
    # Process users in chunks
    valid_user_count = 0
    chunk_size = 500
    
    for chunk_start in range(0, len(user_ids), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(user_ids))
        chunk_users = user_ids[chunk_start:chunk_end]
        
        logger.info(f"Processing chunk {chunk_start//chunk_size + 1}: users {chunk_start} to {chunk_end} ({len(chunk_users)} users)")
        
        # Fetch interactions for this chunk
        chunk_interactions = fetch_interactions_chunk(conn, chunk_users, batch_size)
        
        logger.info(f"Fetched interactions for {len(chunk_interactions)} users in this chunk")
        
        # Process interactions for this chunk
        for user_id in chunk_users:
            if valid_user_count >= num_users:
                break
                
            interactions = chunk_interactions.get(user_id, [])
            if interactions:
                logger.debug(f"User {user_id} has {len(interactions)} interactions")
            
            valid_interactions = process_user_interactions(
                user_id, interactions, max_interactions_per_user, cursor, video_metadata_cache, use_openai_embeddings
            )
            
            if valid_interactions and len(valid_interactions) >= 2:
                interaction_dict[user_id] = valid_interactions
                # Update user_id_map
                if str(user_id) not in user_id_map:
                    user_id_map[str(user_id)] = len(user_id_map) + 1
                # Update video_id_map for interacted videos
                for inter in valid_interactions:
                    video_id = str(inter['video_id'])
                    if video_id not in video_id_map:
                        video_id_map[video_id] = len(video_id_map) + 1
                valid_user_count += 1
                logger.debug(f"Added user {user_id} with {len(valid_interactions)} valid interactions")
        
        # Clear chunk data to free memory
        chunk_interactions.clear()
        gc.collect()
        
        if valid_user_count >= num_users:
            break
    
    # Generate cold-start users and items
    interaction_dict, user_id_map, video_id_map = generate_cold_start_data(
        num_cold_users=num_cold_users,
        num_cold_items=num_cold_items,
        interaction_dict=interaction_dict,
        video_metadata_cache=video_metadata_cache,
        user_id_map=user_id_map,
        video_id_map=video_id_map,
        random_seed=random_seed
    )
    
    # Log final statistics
    logger.info(f"Final dataset: {len(interaction_dict)} unique users")
    interaction_counts = {uid: len(inters) for uid, inters in interaction_dict.items()}
    if interaction_counts:
        logger.info(f"Interaction stats - Min: {min(interaction_counts.values())}, "
                    f"Max: {max(interaction_counts.values())}, "
                    f"Avg: {sum(interaction_counts.values()) / len(interaction_counts):.2f}")
    
    cursor.close()
    conn.close()
    
    return {
        "interaction_dict": interaction_dict,
        "num_users": len(interaction_dict),
        "max_interactions_per_user": max_interactions_per_user,
        "num_cold_users": num_cold_users,
        "num_cold_items": num_cold_items,
        "user_id_map": user_id_map,
        "video_id_map": video_id_map
    }

def fetch_interactions_chunk(conn, user_ids, batch_size):
    """
    Fetch interactions for a chunk of users with proper memory management.
    """
    # Create a regular cursor - pg8000 doesn't support server-side cursors
    cursor = conn.cursor()
    chunk_interactions = {}
    
    try:
        # Process in smaller sub-batches to avoid SQL query size issues
        for i in range(0, len(user_ids), batch_size):
            batch_users = user_ids[i:i + batch_size]
            placeholders = ','.join(['%s'] * len(batch_users))
            
            # Execute the query with regular cursor
            cursor.execute(f"""
                SELECT i.*, vm.*
                FROM interaction_filtered i
                LEFT JOIN video_metadata vm ON i.pid = vm.video_id
                WHERE i.user_id IN ({placeholders})
                ORDER BY i.user_id, i.exposed_time
            """, tuple(batch_users))
            
            # Fetch all rows for this batch
            rows = cursor.fetchall()
            
            # Process the results
            columns = [desc[0] for desc in cursor.description]
            for row in rows:
                interaction_dict = dict(zip(columns, row))
                user_id = interaction_dict['user_id']
                if user_id not in chunk_interactions:
                    chunk_interactions[user_id] = []
                chunk_interactions[user_id].append(interaction_dict)
            
            logger.info(f"Processed sub-batch {i//batch_size + 1} with {len(batch_users)} users")
    
    except Exception as e:
        logger.error(f"Failed to fetch interactions: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        cursor.close()
    
    return chunk_interactions
def main():
    logger.info("Initializing components...")
    faiss_api_url = "http://54.149.116.43:8000/embedding"
    video_feature_manager = VideoFeatureManager()
    
    user_embedding_generator = UserEmbeddingGenerator()
    
    recommender = MerlinRecommender(
        user_embedding_generator=user_embedding_generator,
        video_feature_manager=video_feature_manager,
        dropout=0,
        l2_reg=0
    )
    
    # Check if pre-processed data exists
    processed_data_dir = "."  # Current directory
    force_reprocess = False  # Set to True to force reprocessing
    
    # If force_reprocess is False and processed data exists, it will be loaded instead of regenerated
    if force_reprocess or not all(os.path.exists(os.path.join(processed_data_dir, dir_name)) 
                              for dir_name in ["processed_data_train", "processed_data_valid", "processed_data_test"]):
        logger.info("Preparing training data from scratch...")
        # Increase the data size to ensure enough samples for InBatchSampler
        datasets = prepare_training_data(
            use_openai_embeddings=False,
            num_users=7000,
            max_interactions_per_user=40,
            num_cold_users=0,
            num_cold_items=0,
            batch_size=64,  
            random_seed=42
        )
    else:
        logger.info("Using existing processed data. Creating minimal datasets object...")
        datasets = {
            "interaction_dict": {},  # Empty dict since we'll load from files
            "num_users": 7000,  # These values only used for logging/tracking
            "max_interactions_per_user": 40,
            "num_cold_users": 0,
            "num_cold_items": 0
        }
    
    logger.info("Starting training of all models...")
    recommender.train_model(
        datasets, 
        epochs=30, 
        batch_size=128,
        wandb_project="smokeshow_2tower_modelV2",
        learning_rate=0.0005,
        wandb_entity=None,
        wandb_run_name=None,
        num_users=datasets["num_users"],
        max_interactions_per_user=datasets["max_interactions_per_user"],
        num_cold_users=datasets["num_cold_users"],
        num_cold_items=datasets["num_cold_items"],
        processed_data_dir=processed_data_dir,
        force_reprocess=force_reprocess
    )
    
    logger.info("<<<<<<<<<<<<< Training is completed successfully >>>>>>>>>>>>>>")
if __name__ == "__main__":
    main()