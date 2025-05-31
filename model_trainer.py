import os
import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class PhonemeModelTrainer:
    def __init__(self, config_path, data_dir, checkpoint_path):
        self.config = self.load_config(config_path)
        self.data_dir = data_dir
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.label_to_index = None
        self.index_to_label = None

    def load_config(self, config_path):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def load_data(self):
        features = []
        labels = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith("_events.tsv"):
                    file_path = os.path.join(root, file)
                    df = pd.read_csv(file_path, sep="\t")
                    # Ensure numeric conversion for features
                    numeric_features = df[["onset", "duration", "place", "manner", "voicing"]].apply(pd.to_numeric, errors="coerce").fillna(0)
                    features.append(numeric_features.values)
                    labels.extend(df["phoneme1"].fillna("n/a").values)
        return np.vstack(features), labels

    def preprocess_data(self, features, labels):
        # Compute mean and standard deviation
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        
        # Avoid division by zero by replacing zero std with 1
        std[std == 0] = 1
        
        # Standardize features
        features = (features - mean) / std
        
        unique_labels = list(set(labels))
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        labels_encoded = np.array([self.label_to_index[label] for label in labels])
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels_encoded, dtype=torch.long)

    def build_model(self, input_dim, num_classes):
        class PhonemeRecognitionModel(nn.Module):
            def __init__(self, input_dim, hidden_units_1, hidden_units_2, hidden_units_3, hidden_units_4, dropout_rate, num_classes):
                super(PhonemeRecognitionModel, self).__init__()
                self.fc1 = nn.Linear(input_dim, hidden_units_1)
                self.bn1 = nn.BatchNorm1d(hidden_units_1)
                self.fc2 = nn.Linear(hidden_units_1, hidden_units_2)
                self.bn2 = nn.BatchNorm1d(hidden_units_2)
                self.fc3 = nn.Linear(hidden_units_2, hidden_units_3)
                self.bn3 = nn.BatchNorm1d(hidden_units_3)
                self.fc4 = nn.Linear(hidden_units_3, hidden_units_4)
                self.bn4 = nn.BatchNorm1d(hidden_units_4)
                self.fc5 = nn.Linear(hidden_units_4, num_classes)
                self.dropout = nn.Dropout(dropout_rate)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.bn1(self.fc1(x)))
                x = self.dropout(x)
                x = self.relu(self.bn2(self.fc2(x)))
                x = self.dropout(x)
                x = self.relu(self.bn3(self.fc3(x)))
                x = self.dropout(x)
                x = self.relu(self.bn4(self.fc4(x)))
                x = self.dropout(x)
                x = self.fc5(x)
                return x

        self.model = PhonemeRecognitionModel(
            input_dim=input_dim,
            hidden_units_1=self.config["hidden_units_1"],
            hidden_units_2=self.config["hidden_units_2"],
            hidden_units_3=self.config["hidden_units_3"],
            hidden_units_4=self.config["hidden_units_4"],  # New layer
            dropout_rate=self.config["dropout_rate"],
            num_classes=num_classes
        ).to(self.device)

    def initialize_weights(self):
        def init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.model.apply(init_weights)

    def save_model(self, checkpoint_path):
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

    def load_model(self):
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        self.model.eval()
        print(f"Model loaded from {self.checkpoint_path}")

    def train(self):
        features, labels = self.load_data()
        features_tensor, labels_tensor = self.preprocess_data(features, labels)

        dataset = TensorDataset(features_tensor, labels_tensor)
        train_size = int(self.config["train_split"] * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=self.config["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config["batch_size"], shuffle=False)

        self.build_model(features_tensor.shape[1], len(self.label_to_index))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])

        for epoch in range(self.config["epochs"]):
            self.model.train()
            for batch_features, batch_labels in train_loader:
                batch_features, batch_labels = batch_features.to(self.device), batch_labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Clip gradients
                optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

        self.save_model()

    def train_one_epoch(self, epoch):
        self.model.train()
        train_loader = self.get_train_loader()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config["learning_rate"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Reduce LR every 10 epochs

        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(self.device), batch_labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Clip gradients
            optimizer.step()

        scheduler.step()  # Update learning rate
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    def get_train_loader(self):
        # Load and preprocess data
        features, labels = self.load_data()
        features_tensor, labels_tensor = self.preprocess_data(features, labels)

        # Create dataset and split into training and testing sets
        dataset = TensorDataset(features_tensor, labels_tensor)
        train_size = int(self.config["train_split"] * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, _ = torch.utils.data.random_split(dataset, [train_size, test_size])

        # Create DataLoader for training data
        train_loader = DataLoader(train_dataset, batch_size=self.config["batch_size"], shuffle=True)
        return train_loader

    def evaluate(self, checkpoint_path):
        # Load the model from the specified checkpoint
        self.checkpoint_path = checkpoint_path
        self.load_model()

        # Prepare the test data loader
        features, labels = self.load_data()
        features_tensor, labels_tensor = self.preprocess_data(features, labels)
        dataset = TensorDataset(features_tensor, labels_tensor)
        test_size = int((1 - self.config["train_split"]) * len(dataset))
        _, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - test_size, test_size])
        test_loader = DataLoader(test_dataset, batch_size=self.config["batch_size"], shuffle=False)

        # Evaluate the model
        self.model.eval()
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features, batch_labels = batch_features.to(self.device), batch_labels.to(self.device)
                outputs = self.model(batch_features)
                _, predictions = torch.max(outputs, 1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())

        return all_predictions, all_labels

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

