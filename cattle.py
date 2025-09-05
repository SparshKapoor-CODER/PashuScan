# cattle.py
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

# ====================== CONFIG ======================
DATA_DIR = os.path.abspath("Cattle_all")
LABELS_FILE = "labels.xlsx"
MODEL_PATH = "dog_breed_model.pth"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 500
LR = 1e-4
DEBUG = False
RETRAIN_QUEUE = "retrain_queue.csv"
# ====================================================

class DogBreedDataset(Dataset):
    def __init__(self, df, data_dir, transform, breed_to_idx):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.breed_to_idx = breed_to_idx
        # build id->path map
        self.img_paths = []
        for _, row in self.df.iterrows():
            img_name = row['id'].strip()
            found = None
            for ext in ['.jpg', '.jpeg', '.png']:
                p = os.path.join(self.data_dir, img_name + ext)
                if os.path.exists(p):
                    found = p
                    break
            if found is None:
                # skip rows with missing images
                continue
            self.img_paths.append((found, row['breed'].strip()))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path, breed = self.img_paths[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        label = self.breed_to_idx[breed]
        return image, label

class DogBreedCNN(nn.Module):
    def __init__(self, num_classes):
        super(DogBreedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def load_labels(labels_file):
    if labels_file.lower().endswith('.csv'):
        df = pd.read_csv(labels_file)
    else:
        df = pd.read_excel(labels_file)
    df['id'] = df['id'].astype(str).str.strip()
    df['breed'] = df['breed'].astype(str).str.strip()
    return df

def build_datasets(df):
    # Filter out missing images
    valid_rows = []
    for _, row in df.iterrows():
        img_id = row['id']
        for ext in ['.jpg', '.jpeg', '.png']:
            if os.path.exists(os.path.join(DATA_DIR, img_id + ext)):
                valid_rows.append(row)
                break
    if len(valid_rows) == 0:
        raise ValueError("No valid images found")

    df_valid = pd.DataFrame(valid_rows)
    train_df, test_df = train_test_split(df_valid, test_size=0.1, stratify=df_valid['breed'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['breed'], random_state=42)
    return train_df, val_df, test_df

def incorporate_retrain_queue(train_df):
    """If retrain_queue.csv exists, append those images (if available) into train_df"""
    if not os.path.exists(RETRAIN_QUEUE):
        return train_df
    try:
        dq = pd.read_csv(RETRAIN_QUEUE)
        # only use rows with correct filled label and existing path
        extra_rows = []
        for _, r in dq.iterrows():
            path = r.get('image_path', '')
            correct = r.get('correct', '')
            if pd.isna(path) or path == "":
                continue
            if not os.path.exists(path):
                continue
            if not correct or pd.isna(correct):
                continue
            # we will add a synthetic row with id as absolute path (so dataset resolves it)
            extra_rows.append({'id': path, 'breed': correct})
        if extra_rows:
            extra_df = pd.DataFrame(extra_rows)
            train_df = pd.concat([train_df, extra_df], ignore_index=True)
    except Exception as e:
        print(f"Could not incorporate retrain queue: {e}")
    return train_df

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

def main():
    df = load_labels(LABELS_FILE)
    if DEBUG:
        df = df.sample(min(200, len(df)), random_state=42).reset_index(drop=True)

    train_df, val_df, test_df = build_datasets(df)
    # incorporate retrain queue (if any)
    train_df = incorporate_retrain_queue(train_df)

    all_breeds = sorted(pd.concat([train_df, val_df, test_df])['breed'].unique())
    breed_to_idx = {b: i for i, b in enumerate(all_breeds)}
    num_classes = len(all_breeds)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = DogBreedDataset(train_df, DATA_DIR, transform, breed_to_idx)
    val_dataset = DogBreedDataset(val_df, DATA_DIR, transform, breed_to_idx)
    test_dataset = DogBreedDataset(test_df, DATA_DIR, transform, breed_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DogBreedCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val = -1.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_acc = evaluate(model, train_loader, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader.dataset):.4f} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

        # save best
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved best model to {MODEL_PATH} (val {best_val:.2f}%)")

    # final test
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    test_acc = evaluate(model, test_loader, device)
    print(f"Final Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
