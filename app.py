import streamlit as st
import mido
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile
import os
from tensorflow.keras.layers import Layer, Attention, Dense, LayerNormalization
import base64

class PositionalEncoding(Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model

    def build(self, input_shape):
        super(PositionalEncoding, self).build(input_shape)

    def call(self, inputs):
        # Implementation of the positional encoding logic
        return inputs

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'max_sequence_length': self.max_sequence_length,
            'd_model': self.d_model,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class TransformerBlock(Layer):
    def __init__(self, d_model, n_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.multi_head_attention = Attention(use_scale=True)
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm_1 = LayerNormalization(epsilon=1e-6)

        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(d_model),
        ])
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm_2 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x = self.multi_head_attention([inputs, inputs, inputs])
        x = self.dropout_1(x)
        x = self.layer_norm_1(inputs + x)

        ffn_output = self.ffn(x)
        ffn_output = self.dropout_2(ffn_output)
        return self.layer_norm_2(x + ffn_output)

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Load models
lstm_model = load_model('lstm_music_model.h5')
#gan_generator = load_model('gan_music_generator.h5')
transformer_model = load_model('music_transformer_model.h5',custom_objects={'PositionalEncoding': PositionalEncoding, 'TransformerBlock': TransformerBlock})
def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mid;base64,{b64}" type="audio/mid">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )
# Load MIDI data processing functions
def preprocess_data(data, vocab_size, max_sequence_length):
    processed_data = []
    for sequence in data:
        sequence = sequence[:max_sequence_length]
        processed_sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=max_sequence_length, padding='post')[0]
        processed_data.append(processed_sequence)
    return np.array(processed_data)

# Function to generate music using the LSTM model
def generate_music_lstm(start_sequence, num_steps=100):
    max_sequence_length = 500
    generated_sequence = start_sequence.copy()

    for _ in range(num_steps):
        predictions = lstm_model.predict(np.expand_dims(generated_sequence[-max_sequence_length:], axis=0))
        predicted_note = np.argmax(predictions[:, -1, :])
        generated_sequence = np.append(generated_sequence, predicted_note)

    return generated_sequence

# Function to generate music using the GAN model
def generate_music_gan():
    noise = np.random.normal(0, 1, size=(1, 100))
    generated_sequence = gan_generator.predict(noise)
    return generated_sequence[0]

# Function to generate music using the Transformer model
def generate_music_transformer(start_sequence, num_steps=100):
    max_sequence_length = 500
    generated_sequence = start_sequence.copy()

    for _ in range(num_steps):
        predictions = transformer_model.predict(np.expand_dims(generated_sequence[-max_sequence_length:], axis=0))
        predicted_note = np.argmax(predictions[:, -1, :])
        generated_sequence = np.append(generated_sequence, predicted_note)

    return generated_sequence

# Function to play MIDI file
def play_midi(sequence):
    output_midi_file = mido.MidiFile()
    output_track = mido.MidiTrack()
    output_midi_file.tracks.append(output_track)

    for note_value in sequence:
        output_track.append(mido.Message('note_on', note=note_value, velocity=64, time=100))

    # Play the MIDI file
    with mido.open_output() as port:
        for msg in output_midi_file.play():
            port.send(msg)

# Streamlit UI
st.title("Music Generation App")

uploaded_file = st.file_uploader("Choose a MIDI file", type="mid")

if uploaded_file is not None:
    # Save the uploaded MIDI file to a temporary location
    temp_dir = tempfile.TemporaryDirectory()
    temp_file_path = os.path.join(temp_dir.name, "uploaded_file.mid")
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    # Load the MIDI file using mido.MidiFile
    midi = mido.MidiFile(temp_file_path)
    notes = [note.note for track in midi.tracks for note in track if note.type == 'note_on']
    start_sequence = preprocess_data([notes], 128, 500)[0]

    model_option = st.selectbox("Select a model", ["LSTM", "GAN", "Transformer"])

    if st.button("Generate Music"):
        if model_option == "LSTM":
            generated_sequence = generate_music_lstm(start_sequence)
        elif model_option == "GAN":
            generated_sequence = generate_music_gan()
        elif model_option == "Transformer":
            generated_sequence = generate_music_transformer(start_sequence)

        vocab_size = 128
        output_midi = np.clip(generated_sequence, 0, vocab_size - 1).astype(int)
        output_midi_file = mido.MidiFile()
        output_track = mido.MidiTrack()
        output_midi_file.tracks.append(output_track)
        for note_value in output_midi:
            output_track.append(mido.Message('note_on', note=note_value, velocity=64, time=100))

        if model_option == "LSTM":
            output_midi_file.save('generated_music_LSTM.mid')
        elif model_option == "GAN":
            output_midi_file.save('generated_music_GAN.mid')
        elif model_option == "Transformer":    
            output_midi_file.save('generated_music_Transformers.mid')

        #st.audio('generated_music.mid')
        

        # Display the link to the generated MIDI file
        st.success("Music generated successfully!")
        #st.audio(generated_file_path)

    # Clean up the temporary directory when done
    #temp_dir.cleanup()

        #play_midi(generated_sequence)

