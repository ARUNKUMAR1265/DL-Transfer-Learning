# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
Include the problem statement and Dataset


## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Import required libraries and define image transforms.

### STEP 2: 

Load training and testing datasets using ImageFolder.

### STEP 3: 
Visualize sample images from the dataset.


### STEP 4: 

Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.

### STEP 5: 

Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.

### STEP 6: 

Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.




## PROGRAM

### Name: Arunkumar s

### Register Number: 212224230024

```python
print(f"Total number of test samples: {len(test_dataset)}")
first_image1,label=test_dataset[0]
print("Image shape:",first_image1.shape)

model=models.vgg19(weights=VGG19_Weights.DEFAULT)
model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

# Train the model
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses=[]
    val_losses=[]
    model.train()
    for epoch in range(num_epochs):
        running_loss=0.0
        for images,labels in train_loader:
            images = images.to(device)
            labels = labels.to(device) 
            optimizer.zero_grad()
            outputs=model(images)

            target_labels = labels.unsqueeze(1).float().to(device)
            loss=criterion(outputs,target_labels)

            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        train_losses.append(running_loss/len(train_loader))

        
        model.eval()
        val_loss=0.0
        with torch.no_grad():
          for images,labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs=model(images)
            target_labels = labels.unsqueeze(1).float().to(device)
            loss=criterion(outputs,target_labels)
            val_loss+=loss.item()
        val_losses.append(val_loss/len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name: Arunkuamar")
    print("Register Number:  212224230024 ")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

```


### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot

<img width="687" height="229" alt="image" src="https://github.com/user-attachments/assets/a62a3635-1840-4754-911c-7679dee7b1a5" />
<img width="883" height="730" alt="image" src="https://github.com/user-attachments/assets/4895b95b-eacb-4177-996a-591a9dbdc87f" />


## Confusion Matrix

<img width="836" height="767" alt="image" src="https://github.com/user-attachments/assets/d59a1724-f5dd-4e40-a94e-fd0f0353967f" />


## Classification Report
<img width="564" height="257" alt="image" src="https://github.com/user-attachments/assets/ea18deb8-de21-4756-8daf-b1dd5425f554" />


### New Sample Data Prediction
<img width="549" height="512" alt="image" src="https://github.com/user-attachments/assets/60f0f5d0-0298-4217-82a5-1145c3d4a97c" />
<img width="449" height="493" alt="image" src="https://github.com/user-attachments/assets/2f17a731-a463-45c1-b1f5-9b28e4ae22ff" />


## RESULT
The image classification model using transfer learning with VGG19 architecture for the given dataset has been executed successfully.
