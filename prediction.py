import os
import nltk
from nltk.corpus import cmudict
from itertools import islice
import csv
import torch
from model_trainer import PhonemeModelTrainer  # Make sure this import works

model_base_path = os.path.abspath("..\\Mind-to-Speech\\models")
artifact_model = "model_checkpoint_epoch2_acc0.5073.pth"

def prediction():
    print(f"Using the highest accuracy model: {artifact_model}")
    # TODO: make prediction using the model
    result = ''
    print(f"Generated Phenomes: {result}")
    return result

# nltk.download('cmudict')
cmu_dict = cmudict.dict()

# Invert the dictionary
phoneme_to_word = {}
for word, phoneme_lists in cmu_dict.items():
    for phonemes in phoneme_lists:
        key = ' '.join(phonemes)
        phoneme_to_word.setdefault(key, []).append(word.lower())

def generate_phoneme_chunks(phonemes, max_len=7):
    results = []

    def backtrack(start, path):
        if start == len(phonemes):
            results.append(path)
            return
        for end in range(start + 1, min(len(phonemes) + 1, start + max_len + 1)):
            chunk = ' '.join(phonemes[start:end])
            if chunk in phoneme_to_word:
                backtrack(end, path + [chunk])

    backtrack(0, [])
    return results

def score_sentence(words):
    return len(words)  # You could use a language model here

def phonemes_to_sentence(phoneme_seq_list):
    """
    Converts a list of lists of phonemes into a sentence.
    Each sublist represents phonemes for a single word.
    """
    sentence = []
    for phoneme_seq in phoneme_seq_list:
        # Decode each sublist of phonemes into a word
        chunk = ' '.join(phoneme_seq)
        word_options = phoneme_to_word.get(chunk, [])
        if word_options:
            sentence.append(word_options[0])  # Use the first word option
        else:
            sentence.append(" ")  # Placeholder for unknown phonemes

    # Construct the sentence
    return ' '.join(sentence).capitalize() + '.'




if __name__ == "__main__":
    test_csv_path = os.path.abspath("..\\Mind-to-Speech\\test\\test_5.csv")
    model_path = os.path.abspath("..\\Mind-to-Speech\\models\\cnn_model_checkpoint_epoch92_acc0.7015.pth")

    # Initialize the trainer and load the model
    trainer = PhonemeModelTrainer(
        config_path=os.path.abspath("..\\Mind-to-Speech\\CONFIG.yml"),
        data_dir=os.path.abspath("..\\Mind-to-Speech\\data"),
        checkpoint_path=model_path
    )
    features, labels = trainer.load_data()
    trainer.preprocess_data(features, labels)  # This sets label_to_index

    # Set up model based on architecture
    if trainer.config.get("architecture", "MLP") == "MLP":
        input_dim = features.shape[1] 
    else:  
        input_dim = trainer.config["conv_channels_1"]

    trainer.build_model(input_dim=input_dim, num_classes=len(trainer.label_to_index))
    trainer.model.load_state_dict(torch.load(model_path, map_location=trainer.device))
    trainer.model.eval()

    phoneme_list = []

    import csv
    import numpy as np

    # 1. Read the CSV and group rows into words (split by empty rows)
    with open(test_csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if all(not value.strip() for value in row.values()):
                phoneme_list.append(' ')  # Add a space to denote a new word
                continue

            # feature_columns = ["stimulus", "tms", "tmstarget", "place", "manner", "voicing", "category"]
            feature_columns = ['Phoneme1', 'Phoneme2', 'Phoneme3']
            features = []
            for col in feature_columns:
                val = row.get(col)
                if val and val.strip():
                    features.append(hash(val.strip()) % 1000)  # Use a simple hash for unseen values
                else:
                    features.append(0)
            if not any(features):
                continue

            features_tensor = torch.tensor([features], dtype=torch.float32).to(trainer.device)
            with torch.no_grad():
                output = trainer.model(features_tensor)
                pred_idx = output.argmax(dim=1).item()
                combined_phoneme = trainer.index_to_label.get(pred_idx, "?")
                for p in list(combined_phoneme):
                    phoneme_list.append(p)

    # Split phoneme_list into words at spaces
    words = []
    current_word = []
    for p in phoneme_list:
        if p == ' ':
            if current_word:
                words.append(''.join(current_word))
                current_word = []
        else:
            current_word.append(p)
    if current_word:
        words.append(''.join(current_word))

    print("Phoneme List:", phoneme_list)

    sentence = ' '.join(words).capitalize() + '.'
    print("Sentence:", sentence)

    # # If you want to decode using CMUdict mapping:
    # phoneme_words = [list(word) for word in words]
    # sentence_cmu = phonemes_to_sentence(phoneme_list)
    # print("CMUdict-decoded sentence:", sentence_cmu)

    # Example custom phoneme-to-word mapping
    simple_phoneme_dict = {
        'p uh t': 'put',
        'b ae t': 'bat',
        'd ae d': 'dad',
        't uh b': 'tub',
        # Add more mappings as needed
    }

    def simple_phoneme_decoder(phoneme_seq_list):
        """
        Decodes a list of lists of phonemes into a sentence using a simple mapping.
        If no mapping is found, joins the phonemes as a pseudo-word.
        """
        words = []
        for phoneme_seq in phoneme_seq_list:
            chunk = ' '.join(phoneme_seq)
            word = simple_phoneme_dict.get(chunk)
            if word:
                words.append(word)
            else:
                # Fallback: join phonemes as a pseudo-word
                words.append(''.join(phoneme_seq))
        return ' '.join(words).capitalize() + '.'

    # After building phoneme_words:
    phoneme_words = [list(word) for word in words]
    sentence_simple = simple_phoneme_decoder(phoneme_words)
    print("Simple-decoded sentence:", sentence_simple)