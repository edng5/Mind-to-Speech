import os
import nltk
from nltk.corpus import cmudict
from itertools import islice

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
            sentence.append("[UNKNOWN]")  # Placeholder for unknown phonemes

    # Construct the sentence
    return ' '.join(sentence).capitalize() + '.'




if __name__ == "__main__":
    # Example input: List of lists where each sublist contains phonemes for a word
    phonemes = [['HH', 'AH0', 'L', 'OW1'], ['W', 'ER0', 'L', 'D']]  # Represents "hello world"

    # EEG_readings = ''

    # phenomes = predictionEEG_readings) # TODO: make prediction using the model
    print(phonemes_to_sentence(phonemes))