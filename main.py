def parse_phonemes_to_sentence(self, phonemes):
        words = []
        current_word = []
        for phoneme_idx in phonemes:
            phoneme = self.index_to_label[phoneme_idx]
            if phoneme == " ":
                words.append("".join(current_word))
                current_word = []
            else:
                current_word.append(phoneme)
        if current_word:
            words.append("".join(current_word))
        return " ".join(words)

