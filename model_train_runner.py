from model_trainer import PhonemeModelTrainer
import numpy as np
import os

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

def runner():
    # Paths to configuration, data, and checkpoint files
    config_path = os.path.abspath("..\\Mind-to-Speech\\CONFIG.yml")
    data_dir = os.path.abspath("..\\Mind-to-Speech\\data")
    models_dir = os.path.abspath("..\\Mind-to-Speech\\temp_models")
    base_checkpoint_path = os.path.join(models_dir, "model_checkpoint")

    print(f"Config Path: {config_path}")
    print(f"Data Directory: {data_dir}")
    print(f"Models Directory: {models_dir}")

    # Ensure the models directory exists
    os.makedirs(models_dir, exist_ok=True)

    # Initialize the trainer
    trainer = PhonemeModelTrainer(config_path=config_path, data_dir=data_dir, checkpoint_path=None)

    # Build the model
    print("Building the model...")
    features, labels = trainer.load_data()
    features_tensor, labels_tensor = trainer.preprocess_data(features, labels)
    trainer.build_model(input_dim=features_tensor.shape[1], num_classes=len(trainer.label_to_index))

    # Track the highest accuracy checkpoint
    highest_accuracy = 0.0
    best_checkpoint_path = None

    # Get the save threshold from the configuration
    save_threshold = trainer.config.get("save_threshold", 0.5)

    # Train the model with evaluation after each epoch
    print("Starting training...")
    for epoch in range(trainer.config["epochs"]):
        print(f"Epoch {epoch + 1}...")
        trainer.train_one_epoch(epoch)

        # Save the model checkpoint
        checkpoint_path = f"{base_checkpoint_path}_epoch{epoch + 1}.pth"
        trainer.save_model(checkpoint_path)

        # Evaluate the model
        print("Evaluating the model...")
        predictions, labels = trainer.evaluate(checkpoint_path)

        # Convert predictions and labels to NumPy arrays
        predictions = np.array(predictions)
        labels = np.array(labels)

        # Calculate accuracy
        accuracy = (predictions == labels).mean()
        print(f"Accuracy after epoch {epoch + 1}: {accuracy:.4f}")

        # Save checkpoint only if accuracy > save_threshold
        if accuracy > save_threshold:
            updated_checkpoint_path = f"{base_checkpoint_path}_epoch{epoch + 1}_acc{accuracy:.4f}.pth"
            os.rename(checkpoint_path, updated_checkpoint_path)
            print(f"Checkpoint saved: {updated_checkpoint_path}")

            # Update the highest accuracy checkpoint
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                best_checkpoint_path = updated_checkpoint_path

if __name__ == "__main__":
    runner()