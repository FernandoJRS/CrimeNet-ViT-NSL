import tensorflow as tf
from tensorflow import keras


learning_rate = 0.0001
weight_decay = 0.0001
image_size = 224  # We'll resize input images to this size (The recommended size is 224)
patch_size = 24  # Size of the patches to be extract from the input images (For 224 size the recommended patch size is 24)
num_patches = (image_size // patch_size) ** 2
projection_dim = 16
num_heads = 4
transformer_units = [projection_dim * 2, projection_dim, ]
transformer_layers = 16
mlp_head_units = [2048, 1024]
channels = 3
input_shape = (image_size, image_size, channels)

num_classes = 2

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def create_vit_classifier():
    inputs_1 = tf.keras.layers.Input(shape=input_shape, name='feature_1')
    inputs_2 = tf.keras.layers.Input(shape=input_shape, name='feature_2')
    # Create patches.
    patches_1 = Patches(patch_size)(inputs_1)
    patches_2 = Patches(patch_size)(inputs_2)
    # Encode patches.
    encoded_patches_1 = PatchEncoder(num_patches, projection_dim)(patches_1)
    encoded_patches_2 = PatchEncoder(num_patches, projection_dim)(patches_2)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches_1)

        # Create a multi-head attention layer.
        attention_output_1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1_1, x1_1)
        # Skip connection 1.
        x2_1 = tf.keras.layers.Add()([attention_output_1, encoded_patches_1])
        # Layer normalization 2.
        x3_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2_1)
        # MLP.
        x3_1 = mlp(x3_1, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches_1 = tf.keras.layers.Add()([x3_1, x2_1])
    
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches_2)

        # Create a multi-head attention layer.
        attention_output_2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1_2, x1_2)
        # Skip connection 1.
        x2_2 = tf.keras.layers.Add()([attention_output_2, encoded_patches_2])
        # Layer normalization 2.
        x3_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2_2)
        # MLP.
        x3_2 = mlp(x3_2, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches_2 = tf.keras.layers.Add()([x3_2, x2_2])
    
    # Create a [batch_size, projection_dim] tensor.
    representation_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches_1)
    representation_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches_2)
    representation = tf.keras.layers.Concatenate()([representation_1, representation_2])
    representation = tf.keras.layers.Flatten()(representation)
    representation = tf.keras.layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = tf.keras.layers.Dense(num_classes, activation='softmax')(features)
    # Create the Keras model.
    model = keras.Model(inputs=[inputs_1, inputs_2], outputs=logits)
    return model
