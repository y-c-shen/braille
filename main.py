import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import re
from sklearn.model_selection import train_test_split

# Custom Dataset class for Braille images
class BrailleDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

# CNN Model
class BrailleCNN(nn.Module):
    def __init__(self):
        super(BrailleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 26)  # 26 classes for alphabet
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def prepare_data():
    dataset_path = "Braille Dataset"
    image_paths = []
    labels = []
    
    # Get all image paths and extract labels
    for filename in os.listdir(dataset_path):
        if filename.endswith('.jpg'):
            # Extract the character (first letter of filename)
            char = filename[0]
            # Convert character to label (0-25)
            label = ord(char.lower()) - ord('a')
            
            image_paths.append(os.path.join(dataset_path, filename))
            labels.append(label)
    
    return train_test_split(image_paths, labels, test_size=0.2, random_state=42)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {100.*train_correct/train_total:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {100.*val_correct/val_total:.2f}%')
        print('--------------------')

def predict_image(model, image_path, device):
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    try:
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image = transform(image).unsqueeze(0)  # Add batch dimension
        image = image.to(device)
        
        # Set model to evaluation mode
        model.eval()
        
        # Make prediction
        with torch.no_grad():
            output = model(image)
            _, predicted = output.max(1)
            predicted_char = chr(predicted.item() + ord('a'))
            
            # Get probability distribution
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence = probabilities[predicted].item() * 100
            
        return predicted_char, confidence
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None, None

def test_single_image(image_path):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the trained model
    model = BrailleCNN().to(device)
    try:
        model.load_state_dict(torch.load('braille_cnn.pth', map_location=device))
    except FileNotFoundError:
        print("Error: Trained model file 'braille_cnn.pth' not found. Please train the model first.")
        return
    
    # Make prediction
    predicted_char, confidence = predict_image(model, image_path, device)
    
    if predicted_char is not None:
        print(f"Predicted character: {predicted_char}")
        print(f"Confidence: {confidence:.2f}%")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare data
    train_paths, val_paths, train_labels, val_labels = prepare_data()
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create datasets
    train_dataset = BrailleDataset(train_paths, train_labels, transform)
    val_dataset = BrailleDataset(val_paths, val_labels, transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model, criterion, and optimizer
    model = BrailleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    num_epochs = 20
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
    
    # Save the model
    torch.save(model.state_dict(), 'braille_cnn.pth')

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        if len(sys.argv) != 3:
            print("Usage for testing: python main.py test <image_path>")
        else:
            test_single_image(sys.argv[2])
    else:
        main()
