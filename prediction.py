import os

model_base_path = os.path.abspath("..\\Mind-to-Speech\\models")
artifact_model = "model_checkpoint_epoch2_acc0.5073.pth"

def prediction():
    print(f"Using the highest accuracy model: {artifact_model}")
    # TODO: make prediction using the model
    result = ''
    print(f"Generated Sentence: {result}")

if __name__ == "__main__":
    prediction()