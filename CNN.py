import zipfile
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Step 1: Read CSV data directly from ZIP file without extracting it
def load_data_from_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        data_frames = []
        for filename in zip_ref.namelist():
            if filename.endswith('.csv'):
                with zip_ref.open(filename) as file:
                    df = pd.read_csv(file, delimiter=";")
                    print(f"Columns in {filename}: {df.columns.tolist()}")  # Print column names for debugging
                    data_frames.append(df)
        return pd.concat(data_frames, ignore_index=True)

# Step 2: Preprocess data to ignore bubbles with VeloOut == -1
def preprocess_data(df, num_bins=40):
    df = df[df['VeloOut'] != -1]
    inputs = []
    labels = []
    for _, row in df.iterrows():
        voltage_exit = np.fromstring(row['voltage_exit'][1:-1], sep=',')
        inputs.append(voltage_exit)
        labels.append(row['VeloOut'])

    inputs = np.array(inputs)
    labels = np.array(labels)

    bin_edges = np.linspace(np.min(labels), np.max(labels), num_bins + 1)
    labels_binned = np.digitize(labels, bin_edges) - 1
    labels_binned[labels_binned == num_bins] = num_bins - 1
    
    return inputs, labels_binned

# Step 3: Dataset class for training and evaluation
class BubbleDataset(Dataset):
    def __init__(self, inputs, labels, num_bins=40):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

# Step 4: Define the CNN Model
class BubbleCNN(nn.Module):
    def __init__(self, input_length, num_classes=40):
        super(BubbleCNN, self).__init__()

        # Define the convolutional layer
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Adjust the input size to match the output of Conv1d
        self.fc1 = nn.Linear(32 * input_length, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for Conv1d
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)  # Flatten for fully connected layer
        x = self.fc1(x)  # Logits for classification
        return x

# Step 5: Train and evaluate the model with validation and test sets
def train_and_evaluate(inputs, labels, input_length, num_bins=40, epochs=10, batch_size=32, val_split=0.2, test_split=0.1):
    # Split data into train and test sets
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, labels, test_size=test_split, random_state=42)
    
    # Further split the train set into train and validation sets
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(train_inputs, train_labels, test_size=val_split, random_state=42)

    # Create Datasets
    train_dataset = BubbleDataset(train_inputs, train_labels, num_bins=num_bins)
    val_dataset = BubbleDataset(val_inputs, val_labels, num_bins=num_bins)
    test_dataset = BubbleDataset(test_inputs, test_labels, num_bins=num_bins)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = BubbleCNN(input_length, num_classes=num_bins)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    
    # Train the model
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_inputs, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_inputs, batch_labels in val_loader:
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss / len(train_loader):.4f}, Validation Loss: {val_loss / len(val_loader):.4f}")

    # Test the model
    model.eval()
    predictions = []
    ground_truth = []
    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            outputs = model(batch_inputs)
            predicted_bins = torch.argmax(outputs, axis=1).tolist()
            predictions.extend(predicted_bins)
            ground_truth.extend(batch_labels.tolist())

    cm = confusion_matrix(ground_truth, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(num_bins), yticklabels=np.arange(num_bins))
    plt.xlabel('Predicted Bins')
    plt.ylabel('True Bins')
    plt.title('Confusion Matrix')
    plt.show()

    # Plot histogram of ground truth vs predictions
    plt.figure(figsize=(10, 6))
    plt.hist(ground_truth, bins=np.arange(num_bins + 1) - 0.5, alpha=0.5, label='Ground Truth', color='blue', edgecolor='black')
    plt.hist(predictions, bins=np.arange(num_bins + 1) - 0.5, alpha=0.5, label='Predictions', color='orange', edgecolor='black')
    plt.xlabel('Bin Index')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Ground Truth vs Predictions Histogram')
    plt.grid(True)
    plt.show()

def main(zip_path, num_bins=40, sequence_length=4001, epochs=150, batch_size=32):
    df = load_data_from_zip(zip_path)
    inputs, labels = preprocess_data(df, num_bins)
    train_and_evaluate(inputs, labels, input_length=sequence_length, num_bins=num_bins, epochs=epochs, batch_size=batch_size)

if __name__ == "__main__":
    zip_path = "all_bubbles.zip"
    main(zip_path)


                
# def model_CNN(data=None, epochs=4, batch_size=10, learning_rate=0.001, loss_fn=torch.nn.MSELoss()):
    


#     CNN = nn.Sequential(
#         nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),  
#         nn.ReLU(), 
#         nn.MaxPool1d(kernel_size=2, stride=2), 
#         nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
#         nn.ReLU(),
#         nn.MaxPool1d(kernel_size=2, stride=2),
#         nn.Flatten(),  
#         nn.Linear(32 * (10 // 4), 64), 
#         nn.ReLU(),
#         nn.Linear(64, 1)  
#     )
#     optimizer = torch.optim.Adam(CNN.parameters(), lr=learning_rate)

#     for epoch in range(epochs):
#         epoch_loss = 0
#         CNN.train()
#         for i in range(0, len(X_train), batch_size):
#             input = X_train[i:i+batch_size]
#             targets = y_train[i:i+batch_size]

#             predictions = CNN(input).squeeze()

#             loss = loss_fn(predictions, targets)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             epoch_loss += loss.item()
        
#         print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / (len(X_train) // batch_size):.4f}")

#     return CNN