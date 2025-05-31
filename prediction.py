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

def phonemes_to_sentence(phoneme_seq):
    all_chunks = generate_phoneme_chunks(phoneme_seq)
    best_sentence = []
    best_score = -1

    for chunk_seq in all_chunks:
        word_options = [phoneme_to_word.get(chunk, []) for chunk in chunk_seq]
        if not all(word_options):  # Skip invalid chunks
            continue
        sentence = [options[0] for options in word_options]
        score = score_sentence(sentence)
        if score > best_score:
            best_score = score
            best_sentence = sentence

    if not best_sentence:
        return "No valid sentence could be generated from the given phonemes."
    return ' '.join(best_sentence).capitalize() + '.'




if __name__ == "__main__":
    phonemes = ['HH', 'AH0', 'L', 'OW1'] # temp testing
    # phenomes = prediction() # TODO: make prediction using the model
    print(phonemes_to_sentence(phonemes))