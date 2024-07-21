import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Part 1 - Data Preprocessing

# Define transformations for the training and test sets
train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the datasets
train_dataset = datasets.ImageFolder('dataset/training_set', transform=train_transforms)
test_dataset = datasets.ImageFolder('dataset/test_set', transform=test_transforms)

# Define the data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Part 2 - Building the CNN

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=32 * 16 * 16, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Instantiate the CNN
cnn = CNN().to(device)

# Part 3 - Training the CNN

# Define the loss function and the optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)

# Training the model
epochs = 25
for epoch in range(epochs):
    cnn.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float().view(-1, 1)
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

# Part 4 - Making a single prediction

def predict_image(image_path, model):
    model.eval()
    img = Image.open(image_path)
    img = test_transforms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
    return 'dog' if output.item() >= 0.5 else 'cat'

# Prediction
prediction = predict_image('dataset/single_prediction/cat_or_dog_1.jpg', cnn)
print(prediction)
