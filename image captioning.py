import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np

# Step 1: Feature Extraction with ResNet50
def extract_features(image_path):
    # Load the pre-trained ResNet50 model + higher level layers
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    model = Model(inputs=base_model.input, outputs=base_model.output)

    # Load and preprocess the image
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # Extract features
    features = model.predict(image)
    return features

# Step 2: Caption Generation with LSTM
class CaptioningModel(tf.keras.Model):
    def __init__(self, vocab_size, embed_size, units):
        super(CaptioningModel, self).__init__()
        self.embedding = Embedding(vocab_size, embed_size)
        self.lstm = LSTM(units, return_sequences=True, return_state=True)
        self.dense1 = Dense(units, activation='relu')
        self.dense2 = Dense(vocab_size)

    def call(self, features, captions):
        # Create embeddings for the captions
        caption_embeddings = self.embedding(captions)
        # Combine features and caption embeddings
        features_expanded = tf.expand_dims(features, 1)
        combined = tf.concat([features_expanded, caption_embeddings], axis=1)
        # Pass through LSTM
        lstm_output, _, _ = self.lstm(combined)
        # Pass through dense layers
        x = self.dense1(lstm_output)
        x = self.dense2(x)
        return x

    def generate_caption(self, features, tokenizer, max_length):
        input_seq = [tokenizer.word_index['<start>']]
        result = []

        for _ in range(max_length):
            sequence = pad_sequences([input_seq], maxlen=max_length, padding='post')
            predictions = self.call(features, sequence)
            predicted_id = tf.argmax(predictions[0, -1, :]).numpy()
            input_seq.append(predicted_id)

            if tokenizer.index_word.get(predicted_id) == '<end>':
                break

            result.append(tokenizer.index_word.get(predicted_id, ''))

        return ' '.join(result)

# Load your image
image_path = 'path/to/your/image.jpg'  # Replace with your image path
features = extract_features(image_path)

# Step 3: Define a simple tokenizer
# Assume a small predefined vocabulary for demonstration purposes
vocab = {'<start>': 1, 'a': 2, 'man': 3, 'with': 4, 'hat': 5, '<end>': 6}
vocab_size = len(vocab) + 1  # +1 for padding token
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.word_index = vocab
tokenizer.index_word = {v: k for k, v in vocab.items()}  # Reverse mapping

# Model parameters
embed_size = 256
units = 512
max_length = 20

# Instantiate the captioning model
captioning_model = CaptioningModel(vocab_size, embed_size, units)

# Generate a caption for the image
features = np.reshape(features, (1, -1))  # Ensure features are in correct shape
caption = captioning_model.generate_caption(features, tokenizer, max_length)
print(f'Generated Caption: {caption}')
