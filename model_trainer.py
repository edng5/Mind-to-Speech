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

        # Directories to load data from
        # directories = [self.data_dir, os.path.join(os.path.dirname(self.data_dir), "augmented_data")]
        directories = [self.data_dir]
        
        for directory in directories:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith(".csv"):
                        file_path = os.path.join(root, file)
                        df = pd.read_csv(file_path)

                        # Skip empty DataFrames
                        if df.empty:
                            print(f"File {file_path} is empty. Skipping...")
                            continue

                        # Normalize column names
                        df.columns = df.columns.str.strip().str.lower()  # Strip spaces and convert to lowercase

                        # Extract specified features
                        feature_columns = ["stimulus", "tms", "tmstarget", "place", "manner", "voicing", "category"]
                        if not all(col in df.columns for col in feature_columns):
                            raise KeyError(f"Missing columns in {file_path}: {set(feature_columns) - set(df.columns)}")

                        extracted_features = df[feature_columns].apply(lambda x: pd.factorize(x)[0], axis=0)  # Factorize categorical features
                        features.append(extracted_features.values)

                        # Combine Phoneme1 and Phoneme2 into a single column
                        df["combined_phonemes"] = df["phoneme1"].fillna("") + "" + df["phoneme2"].fillna("")
                        combined_labels = pd.factorize(df["combined_phonemes"])[0]  # Factorize combined phonemes
                        labels.append(combined_labels)

        return np.vstack(features), np.hstack(labels)

    def preprocess_data(self, features, labels):
        # Standardize features
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        features = (features - mean) / std

        # Reshape features for Transformer architecture
        if self.config["architecture"] == "Transformer":
            # Ensure features are reshaped to (batch_size, seq_len, input_dim)
            input_dim = features.shape[1]  # Use the number of features as input dimension
            seq_len = features.shape[0]  # Treat each sample as a sequence
            features = features.reshape(seq_len, 1, input_dim)

        # Encode labels
        unique_labels = np.unique(labels)
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        labels_encoded = np.array([self.label_to_index[label] for label in labels])  # Handle 1D labels

        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels_encoded, dtype=torch.long)

    def build_model(self, input_dim, num_classes):
        architecture = self.config["architecture"]

        if architecture == "CNN":
            self.build_cnn(input_dim, num_classes)
        elif architecture == "MLP":
            self.build_mlp(input_dim, num_classes)
        elif architecture == "Transformer":
            self.build_transformer(input_dim, num_classes)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    def build_cnn(self, input_dim, num_classes):
        class PhonemeRecognitionCNN(nn.Module):
            def __init__(self, input_dim, conv_channels, linear_layers, dropout_rate, num_classes):
                super(PhonemeRecognitionCNN, self).__init__()
                self.conv_layers = nn.ModuleList()
                self.num_conv_layers = len(conv_channels)

                # Create convolutional layers dynamically
                for i in range(self.num_conv_layers):
                    in_channels = 1 if i == 0 else conv_channels[i - 1]
                    out_channels = conv_channels[i]
                    self.conv_layers.append(nn.Sequential(
                        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU()
                    ))

                # Global average pooling
                self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

                # Fully connected layers
                self.fc_layers = nn.ModuleList()
                for i in range(len(linear_layers)):
                    in_features = conv_channels[-1] if i == 0 else linear_layers[i - 1]
                    out_features = linear_layers[i]
                    self.fc_layers.append(nn.Sequential(
                        nn.Linear(in_features, out_features),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate)
                    ))

                # Output layer
                self.output_layer = nn.Linear(linear_layers[-1], num_classes)

            def forward(self, x):
                x = x.unsqueeze(1)  # Add channel dimension for Conv1D
                for conv_layer in self.conv_layers:
                    x = conv_layer(x)
                x = self.global_avg_pool(x)  # Apply global average pooling
                x = x.view(x.size(0), -1)  # Flatten for fully connected layers
                for fc_layer in self.fc_layers:
                    x = fc_layer(x)
                x = self.output_layer(x)  # Output shape: (batch_size, num_classes)
                return x

        conv_channels = [
            self.config[f"conv_channels_{i}"] for i in range(1, 6)
        ]
        linear_layers = [
            self.config[f"linear_layer_{i}"] for i in range(1, 3)
        ]

        self.model = PhonemeRecognitionCNN(
            input_dim=input_dim,
            conv_channels=conv_channels,
            linear_layers=linear_layers,
            dropout_rate=self.config["dropout_rate"],
            num_classes=num_classes
        ).to(self.device)

    def build_mlp(self, input_dim, num_classes):
        class PhonemeRecognitionMLP(nn.Module):
            def __init__(self, input_dim, hidden_units, dropout_rate, num_classes):
                super(PhonemeRecognitionMLP, self).__init__()
                self.layers = nn.ModuleList()
                self.num_layers = len(hidden_units)

                # Create layers dynamically
                for i in range(self.num_layers):
                    in_features = input_dim if i == 0 else hidden_units[i - 1]
                    out_features = hidden_units[i]
                    self.layers.append(nn.Sequential(
                        nn.Linear(in_features, out_features),
                        nn.BatchNorm1d(out_features),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate)
                    ))

                # Output layer
                self.output_layer = nn.Linear(hidden_units[-1], num_classes)

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                x = self.output_layer(x)  # Output shape: (batch_size, num_classes)
                return x

        hidden_units = [
            self.config[f"hidden_units_{i}"] for i in range(1, 8)
        ]

        self.model = PhonemeRecognitionMLP(
            input_dim=input_dim,
            hidden_units=hidden_units,
            dropout_rate=self.config["mlp_dropout_rate"],
            num_classes=num_classes
        ).to(self.device)

    def build_transformer(self, input_dim, num_classes):
        class PhonemeRecognitionTransformer(nn.Module):
            def __init__(self, input_dim, num_heads, num_layers, hidden_dim, dropout_rate, num_classes):
                super(PhonemeRecognitionTransformer, self).__init__()
                self.embedding = nn.Linear(input_dim, hidden_dim)
                self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, hidden_dim))  # Max sequence length: 1000
                self.transformer = nn.Transformer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    num_encoder_layers=num_layers,
                    num_decoder_layers=num_layers,
                    dropout=dropout_rate
                )
                self.fc = nn.Linear(hidden_dim, num_classes)

            def forward(self, x):
                # Ensure input tensor is shaped as (batch_size, seq_len, input_dim)
                if len(x.shape) != 3:
                    raise ValueError(f"Input tensor must have shape (batch_size, seq_len, input_dim), but got {x.shape}")

                seq_len = x.size(1)
                if seq_len > self.positional_encoding.size(1):
                    raise ValueError(f"Input sequence length ({seq_len}) exceeds maximum positional encoding length ({self.positional_encoding.size(1)}).")

                # Apply embedding and positional encoding
                x = self.embedding(x) + self.positional_encoding[:, :seq_len, :]
                x = self.transformer(x, x)  # Self-attention
                x = x.mean(dim=1)  # Global average pooling
                x = self.fc(x)
                return x

        self.model = PhonemeRecognitionTransformer(
            input_dim=input_dim,  # Match the reshaped input tensor's last dimension
            num_heads=self.config["num_heads"],
            num_layers=self.config["num_layers"],
            hidden_dim=self.config["hidden_dim"],
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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(self.device), batch_labels.to(self.device)

            # Reshape input tensor for Transformer architecture
            if self.config["architecture"] == "Transformer":
                batch_features = batch_features.view(batch_features.size(0), -1, batch_features.size(1))  # Reshape to (batch_size, seq_len, input_dim)

            optimizer.zero_grad()
            outputs = self.model(batch_features)  # Shape: (batch_size, num_classes)
            loss = criterion(outputs, batch_labels)  # Single-column labels
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
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
        if self.config["architecture"] == "SVM":
            # SVM evaluation logic
            features, labels = self.load_data()
            features_tensor, labels_tensor = self.preprocess_data(features, labels)
            predictions = self.model.predict(features_tensor.cpu().numpy())
            return predictions, labels_tensor.cpu().numpy()
        else:
            # Neural network evaluation logic
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
                    outputs = self.model(batch_features)  # Shape: (batch_size, num_classes)
                    _, predictions = torch.max(outputs, 1)  # Get predictions
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(batch_labels.cpu().numpy())

            return np.array(all_predictions), np.array(all_labels)

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

