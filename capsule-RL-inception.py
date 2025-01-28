import os
import torch
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ProgressBar
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from torchmetrics.classification import (
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score
)
from gym import spaces
import gym
import torch.optim as optim
import seaborn as sns
from collections import Counter
from torchsummary import summary
from matplotlib.patches import FancyBboxPatch, FancyArrow

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight  

    def forward(self, inputs, targets):
        if self.weight is not None:
            self.weight = self.weight.to(inputs.device)
        
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

class CapsuleLayer(nn.Module):
    def __init__(self, in_capsules, in_dim, out_capsules, out_dim, num_routes=3):
        super(CapsuleLayer, self).__init__()
        self.in_capsules = in_capsules
        self.in_dim = in_dim
        self.out_capsules = out_capsules
        self.out_dim = out_dim
        self.num_routes = num_routes

        self.W = nn.Parameter(torch.randn(1, in_capsules, out_capsules, out_dim, in_dim))

    def forward(self, x):
        batch_size = x.size(0)

        x = x[:, :, None, :, None] 
        W = self.W.expand(batch_size, -1, -1, -1, -1)  

        u_hat = torch.matmul(W, x)  
        u_hat = u_hat.squeeze(-1)   

        b_ij = torch.zeros(batch_size, self.in_capsules, self.out_capsules).to(x.device)

        for _ in range(self.num_routes):
            c_ij = F.softmax(b_ij, dim=2)  

            s_j = (c_ij.unsqueeze(-1) * u_hat).sum(dim=1)  

            v_j = self.squash(s_j) 

            v_j_expanded = v_j.unsqueeze(1)  
            agreement = (u_hat * v_j_expanded).sum(dim=-1)  

            b_ij = b_ij + agreement

        return v_j  

    def squash(self, s_j):
        s_j_norm = torch.norm(s_j, dim=-1, keepdim=True)
        scale = (s_j_norm ** 2) / (1 + s_j_norm ** 2)
        v_j = scale * (s_j / (s_j_norm + 1e-8))  
        return v_j

class CapsuleRoutingEnv(gym.Env):
    def __init__(self, in_capsules, out_capsules, num_routes):
        super(CapsuleRoutingEnv, self).__init__()
        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        self.num_routes = num_routes

        self.action_space = spaces.Discrete(self.out_capsules)
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(in_capsules,))

    def reset(self):
        self.state = np.random.randn(self.in_capsules)  
        return self.state

    def step(self, action):
        reward = self.compute_reward(action)
        self.state = np.random.randn(self.in_capsules)  
        done = True  
        return self.state, reward, done, {}

    def compute_reward(self, action):
        return np.random.random()  
    
class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        attn_weights = self.attention(x)
        return x * attn_weights

class DQNAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class HybridCNN(pl.LightningModule):
    def __init__(self, agent):
        super(HybridCNN, self).__init__()

        # Inicjalizacja ResNet50
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet_out_features = self.resnet.fc.in_features  # 2048 dla ResNet50
        self.resnet.fc = nn.Identity()

        self.in_capsules = 64
        self.in_dim = 32
        self.out_capsules = 10
        self.out_dim = 16

        self.fc_transform = nn.Linear(self.resnet_out_features, self.in_capsules * self.in_dim)
        self.capsule_layer = CapsuleLayer(
            in_capsules=self.in_capsules,
            in_dim=self.in_dim,
            out_capsules=self.out_capsules,
            out_dim=self.out_dim,
            num_routes=3
        )

        total_features = self.resnet_out_features + self.out_capsules * self.out_dim
        self.fc1 = nn.Linear(total_features, 512)
        self.fc2 = nn.Linear(512, 4)
        self.dropout = nn.Dropout(0.5)

        self.criterion = FocalLoss(alpha=1, gamma=2, weight=torch.tensor([1.0, 7.0, 1.0, 2.0]))

        self.device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = nn.ModuleDict({
            'val_precision': MulticlassPrecision(num_classes=4, average='macro').to(self.device_type),
            'val_recall': MulticlassRecall(num_classes=4, average='macro').to(self.device_type),
            'val_f1': MulticlassF1Score(num_classes=4, average='macro').to(self.device_type),
        })

        self.validation_preds = []
        self.validation_labels = []
        self.validation_features = []
        
    def forward(self, x):
        resnet_output = self.resnet(x) 

        transformed_output = self.fc_transform(resnet_output)
        capsule_input = transformed_output.view(x.size(0), self.in_capsules, self.in_dim)

        capsule_output = self.capsule_layer(capsule_input)
        capsule_output_flat = capsule_output.view(x.size(0), -1)

        combined_features = torch.cat((resnet_output, capsule_output_flat), dim=1)
        combined_features = self.dropout(combined_features)

        x = self.fc1(combined_features)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        preds = torch.argmax(outputs, dim=1)
    
        self.validation_preds.append(preds.cpu())
        self.validation_labels.append(labels.cpu())
    
        f1_score = self.metrics['val_f1'](preds, labels)
        precision = self.metrics['val_precision'](preds, labels)
        self.log('val_f1_step', f1_score, prog_bar=True, on_step=True)
        self.log('val_precision_step', precision, prog_bar=True, on_step=True)
    
        return {"preds": preds, "labels": labels}


    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.validation_preds, dim=0).to(self.device_type)
        all_labels = torch.cat(self.validation_labels, dim=0).to(self.device_type)
    
        # Obliczanie metryk
        f1_score = self.metrics['val_f1'](all_preds, all_labels)
        precision = self.metrics['val_precision'](all_preds, all_labels)
    
        # Logowanie metryk
        self.log('val_f1', f1_score, prog_bar=True, on_epoch=True)
        self.log('val_precision', precision, prog_bar=True, on_epoch=True)
    
        self.validation_preds = []
        self.validation_labels = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]

    def on_train_epoch_end(self):
        self.validation_preds = []
        self.validation_labels = []
        self.validation_features = []

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path, batch_size=64):
        super(CustomDataModule, self).__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Dataset attributes
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if not self.train_dataset or not self.val_dataset:
            dataset = ImageFolder(root=self.dataset_path, transform=self.transform)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])
            print(f"Train dataset length: {len(self.train_dataset)}")
            print(f"Validation dataset length: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True)

def generate_classification_images(val_loader, model):
    batch = next(iter(val_loader))
    images, labels = batch
    images = images.to(model.device)
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    images = images * torch.tensor(std).view(1, 3, 1, 1).to(images.device) + torch.tensor(mean).view(1, 3, 1, 1).to(images.device)
    images = torch.clamp(images, 0, 1)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    
    num_images = min(len(images), 8)  
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))  
    axes = axes.flatten() 
    
    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images[i])
        ax.set_title(f'Predicted: {preds[i].item()}, True: {labels[i].item()}')
        ax.axis('off')
    
    for ax in axes[num_images:]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def generate_individual_classification_images(val_loader, model):
    class_labels = ['Healthy', 'Mild', 'Moderate', 'Severe']

    batch = next(iter(val_loader))
    images, labels = batch
    images = images.to(model.device)
    model.eval()
    
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    images = images * torch.tensor(std).view(1, 3, 1, 1).to(images.device) + torch.tensor(mean).view(1, 3, 1, 1).to(images.device)
    images = torch.clamp(images, 0, 1)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    
    for i in range(len(images)):
        plt.figure(figsize=(4, 4))  
        plt.imshow(images[i])
        plt.title(f'Predicted: {preds[i].item()}, True: {labels[i].item()}')
        plt.axis('off')

        plt.savefig(f"individual_image_{i+1}.png")
        plt.close()

    print(f"Images saved as individual_image_1.png, individual_image_2.png, ..., up to {len(images)} images.")

    """
    Generate individual classification images and save them as separate files.
    
    Parameters:
        val_loader: DataLoader
            Validation data loader.
        model: nn.Module
            Trained model for classification.
        class_labels: list
            List of class labels for predictions, e.g., ['Healthy', 'Mild', 'Moderate', 'Severe'].
    """
    batch = next(iter(val_loader))
    images, labels = batch
    images = images.to(model.device)
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    images = images * torch.tensor(std).view(1, 3, 1, 1).to(images.device) + torch.tensor(mean).view(1, 3, 1, 1).to(images.device)
    images = torch.clamp(images, 0, 1)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    
    num_images = min(len(images), 4) 
    
    for i in range(num_images):
        plt.figure(figsize=(4, 4)) 
        plt.imshow(images[i])
        plt.title(f'Predicted: {class_labels[preds[i].item()]}, True: {class_labels[labels[i].item()]}')
        plt.axis('off')
        
        plt.savefig(f'image_{i+1}.png', bbox_inches='tight')
        plt.close() 

    print(f"Saved {num_images} individual images as PNG files.")

    """
    Generate individual classification images and save them as separate files.
    
    Parameters:
        val_loader: DataLoader
            Validation data loader.
        model: nn.Module
            Trained model for classification.
        class_labels: list
            List of class labels for predictions, e.g., ['Healthy', 'Mild', 'Moderate', 'Severe'].
    """
    batch = next(iter(val_loader))
    images, labels = batch
    images = images.to(model.device)
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    images = images * torch.tensor(std).view(1, 3, 1, 1).to(images.device) + torch.tensor(mean).view(1, 3, 1, 1).to(images.device)
    images = torch.clamp(images, 0, 1)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    
    num_images = min(len(images), 4) 
    
    for i in range(num_images):
        plt.figure(figsize=(4, 4)) 
        plt.imshow(images[i])
        plt.title(f'Predicted: {class_labels[preds[i].item()]}, True: {class_labels[labels[i].item()]}')
        plt.axis('off')
        
        # Save image as PNG file
        plt.savefig(f'image_{i+1}.png', bbox_inches='tight')
        plt.close() 

    print(f"Saved {num_images} individual images as PNG files.")
    
def collect_predictions(model, dataloader, device):
    """Zbierz predykcje i prawdziwe etykiety z całego zbioru walidacyjnego."""
    model.to(device) 
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)  
            labels = labels.to(device)  
            outputs = model(inputs)  
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy()) 
            all_labels.extend(labels.cpu().numpy()) 

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(y_true, y_pred, class_labels):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_labels)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == '__main__':
    class_labels = ['Class1', 'Class2', 'Class3', 'Class4']

    
    dataset_path = 'D:/Badania/embedded/Dataset'
    data_module = CustomDataModule(dataset_path)

    env = CapsuleRoutingEnv(in_capsules=64, out_capsules=10, num_routes=3)
    state_dim = env.observation_space.shape[0]
    
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)

    model = HybridCNN(agent=agent)
    model.to('cuda')
    
    summary(model, (3, 299, 299))
    
    
    logger = TensorBoardLogger('tb_logs', name='cap-agents')
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", save_top_k=1, verbose=True)
    progress_bar = ProgressBar()
    
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='gpu',
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, progress_bar]
    )
     
    trainer.fit(model, datamodule=data_module)

    generate_classification_images(data_module.val_dataloader(), model)
    
    generate_individual_classification_images(data_module.val_dataloader(), model)
    
    val_loader = data_module.val_dataloader() 
    preds, labels = collect_predictions(model, val_loader, device='cuda')
    
    plot_confusion_matrix(labels, preds, class_labels)