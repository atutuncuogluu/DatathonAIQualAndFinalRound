##############################################
# TEST/INFERENCE KODU (Z-SCORE + MLP’li ViT)
##############################################

import os
import math
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# transformers kütüphanesi
from transformers import ViTModel
from tqdm import tqdm

# =================================
# 1. DOSYA YOLLARI & Z-SCORE DEĞERLERİ
# =================================
TEST_CSV  = "/kaggle/input/datathon-ai-24/test.csv"
TEST_DIR  = "/kaggle/input/datathon-ai-24/test"

# Eğitimde kullanılan aynı Z-Score istatistikleri:
LAT_MEAN = 41.105763
LAT_STD  = 0.002358
LON_MEAN = 29.025191
LON_STD  = 0.004205

def inverse_zscore_lat_lon(lat_norm, lon_norm):
    lat_real = lat_norm * LAT_STD + LAT_MEAN
    lon_real = lon_norm * LON_STD + LON_MEAN
    return lat_real, lon_real

# =================================
# 2. TEST CSV OKUMA
# =================================
print("Test CSV okunuyor...")
test_df = pd.read_csv(TEST_CSV, sep=';')  # 'filename' sütunu olduğu varsayılır

# =================================
# 3. TEST DATASET TANIMI
# =================================
class TestDataset(Dataset):
    """
    Sadece resim yolunu alır, transform uygular.
    Modelin tahminiyle latitude/longitude hesaplamak için
    Z-Score inverse fonksiyonu ayrı yapılacak.
    """
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = os.path.join(self.img_dir, row["filename"])
       
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
       
        return image

# =================================
# 4. DATA TRANSFORMS ve DataLoader
# =================================
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Aynı ImageNet normalizasyonu (train ile tutarlı)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

test_dataset = TestDataset(
    dataframe=test_df,
    img_dir=TEST_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4
)

# =======================================
# 5. MODEL TANIMI (MLP’li ViT)
# =======================================
# Eğitimde kullandığınız MLP’li yapıyı burada tanımlayın:
class ViTRegressor(nn.Module):
    def __init__(self, pretrained_model_name="google/vit-base-patch16-224-in21k", output_dim=2):
        super().__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        # Özel MLP kafası (örnek)
        self.head = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        outputs = self.vit(x)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        preds = self.head(cls_embedding)   # [batch_size, 2]
        return preds

# =======================================
# 5.1. MODELİ YÜKLEME
# =======================================
# Eğer aynı oturumda iseniz, "model" değişkeni zaten RAM’de vardır.
# Yoksa aşağıdaki gibi yeniden oluşturup state_dict() yükleyebilirsiniz:

# model = ViTRegressor("google/vit-base-patch16-224-in21k", output_dim=2)
# model.load_state_dict(torch.load("/kaggle/working/vit_zscore_haversine.pth"))
# model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# ==============================
# 6. TAHMİN (INFERENCE)
# ==============================
print("Test verisi üzerinde tahminler yapılıyor...")
predictions_lat = []
predictions_lon = []

with torch.no_grad():
    for images in tqdm(test_loader, desc="Inference", unit="batch"):
        images = images.to(device)
       
        # Model çıktısı -> [batch_size, 2] (lat_norm, lon_norm)
        outputs = model(images)
       
        lat_norm = outputs[:, 0]
        lon_norm = outputs[:, 1]
       
        # Ters dönüşüm (Z-Score -> Orijinal)
        lat_vals, lon_vals = inverse_zscore_lat_lon(lat_norm, lon_norm)
       
        # CPU'ya alarak Numpy array'e dönüştür
        lat_vals = lat_vals.cpu().numpy()
        lon_vals = lon_vals.cpu().numpy()
       
        # Listeye ekle
        predictions_lat.extend(lat_vals)
        predictions_lon.extend(lon_vals)

# =================================
# 7. test.csv’Yİ GÜNCELLE
# =================================
test_df["latitude"] = predictions_lat
test_df["longitude"] = predictions_lon

output_path = "/kaggle/working/test_predictions_zscore_vit.csv"
test_df.to_csv(output_path, index=False)

print(f"İşlem tamamlandı! Tahmin sonuçları '{output_path}' dosyasına kaydedildi.")

Ahmet Tütüncüoğlu <atutuncuoglu0@gmail.com>, 21 Ara 2024 Cmt, 18:56 tarihinde şunu yazdı:
import os
import pandas as pd
import numpy as np
from PIL import Image
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTModel
from tqdm import tqdm

# ================================================
# 1) PARAMETRELER ve Z-SCORE NORMALİZASYON
# ================================================
CSV_FILE  = "/kaggle/input/datathon-ai-24/train.csv"
IMG_DIR   = "/kaggle/input/datathon-ai-24/train"

# Örnek Z-Score değerleri (sizin dataset analizinize göre güncelleyin):
LAT_MEAN = 41.105763
LAT_STD  = 0.002358
LON_MEAN = 29.025191
LON_STD  = 0.004205

BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4  # Transformer için görece düşük LR
WARMUP_EPOCHS = 2     # CosineAnnealingLR'e geçmeden önce

# Dünya yarıçapı (km). Haversine için kullanıyoruz.
EARTH_RADIUS = 6371.0

def zscore_lat_lon(lat, lon):
    lat_norm = (lat - LAT_MEAN) / LAT_STD
    lon_norm = (lon - LON_MEAN) / LON_STD
    return lat_norm, lon_norm

def inverse_zscore_lat_lon(lat_norm, lon_norm):
    lat_real = lat_norm * LAT_STD + LAT_MEAN
    lon_real = lon_norm * LON_STD + LON_MEAN
    return lat_real, lon_real

# ================================================
# 2) HAVERSINE METRİĞİ (KM CİNSİNDEN)
# ================================================
def haversine_distance(lat1, lon1, lat2, lon2, radius=EARTH_RADIUS):
    """
    lat1, lon1, lat2, lon2 -> Derece cinsinden
    radius                -> Dünya yarıçapı (km)
   
    Formül:
    d = 2 * r * asin( sqrt( sin^2((lat2-lat1)/2) + cos(lat1)*cos(lat2)*sin^2((lon2-lon1)/2) ) )
    """
    # Dereceleri radyana çevir
    lat1, lon1 = map(math.radians, [lat1, lon1])
    lat2, lon2 = map(math.radians, [lat2, lon2])
   
    dlat = lat2 - lat1
    dlon = lon2 - lon1
   
    a = (math.sin(dlat / 2))**2 + math.cos(lat1) * math.cos(lat2) * (math.sin(dlon / 2))**2
    c = 2 * math.asin(math.sqrt(a))
   
    distance = radius * c  # km
    return distance

def batch_haversine_metric(preds, targets):
    """
    preds, targets -> [batch_size, 2], Z-Score normalized
    1) Ters dönüşüm: (lat_norm -> lat_real), (lon_norm -> lon_real)
    2) Haversine ile her örneğin mesafesini hesaplar, ortalama döndürür (km)
    """
    batch_size = preds.size(0)
    total_dist = 0.0
   
    for i in range(batch_size):
        lat_norm_pred, lon_norm_pred = preds[i]
        lat_norm_true, lon_norm_true = targets[i]
       
        # Ters dönüşüm
        lat_pred, lon_pred = inverse_zscore_lat_lon(lat_norm_pred.item(), lon_norm_pred.item())
        lat_true, lon_true = inverse_zscore_lat_lon(lat_norm_true.item(), lon_norm_true.item())
       
        dist_km = haversine_distance(lat_pred, lon_pred, lat_true, lon_true, EARTH_RADIUS)
        total_dist += dist_km
   
    return total_dist / batch_size  # ortalama mesafe

# ================================================
# 3) DATASET SINIFI
# ================================================
class CampusDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, use_zscore=True):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.use_zscore = use_zscore

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
       
        image_path = os.path.join(self.img_dir, row["filename"])
        lat = float(row["latitude"])
        lon = float(row["longitude"])
       
        image = Image.open(image_path).convert("RGB")
       
        if self.transform:
            image = self.transform(image)
       
        if self.use_zscore:
            lat, lon = zscore_lat_lon(lat, lon)
       
        target = torch.tensor([lat, lon], dtype=torch.float32)
        return image, target

# ================================================
# 4) MODEL: ViT + MLP HEAD
# ================================================
class ViTRegressor(nn.Module):
    def __init__(self, pretrained_model_name="google/vit-base-patch16-224-in21k", hidden_dim=256, output_dim=2):
        super().__init__()
        # 1) Pretrained ViT
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
       
        # 2) Küçük bir MLP çıkış katmanı
        #    vit.config.hidden_size -> 768 (ViT-B için)
        self.mlp_head = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        outputs = self.vit(x)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        preds = self.mlp_head(cls_embedding)  # (batch_size, 2)
        return preds

# ================================================
# 5) EĞİTİM KURULUMU
# ================================================
def main():
    # 5.1 Veri Okuma
    print("CSV dosyası okunuyor...")
    df = pd.read_csv(CSV_FILE, sep=';')
   
    # 5.2 Transform
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
   
    # 5.3 Dataset & DataLoader
    train_dataset = CampusDataset(df, IMG_DIR, transform=train_transform, use_zscore=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
   
    # 5.4 Model
    print("Model oluşturuluyor...")
    model = ViTRegressor(pretrained_model_name="google/vit-base-patch16-224-in21k", hidden_dim=256, output_dim=2)
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
   
    # Çoklu GPU
    if torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()} GPU bulundu! DataParallel kullanılıyor.")
        model = nn.DataParallel(model)
   
    # 5.5 Kaynaklar
    criterion = nn.MSELoss()     # MSE kaybı
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
   
    # 5.6 Scheduler: CosineAnnealingLR (örnek)
    # (İlk WARMUP_EPOCHS sabit LR, sonra cos. schedule)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(NUM_EPOCHS - WARMUP_EPOCHS),
        eta_min=1e-6
    )
   
    # ========================================
    # 6) EĞİTİM DÖNGÜSÜ
    # ========================================
    print("Eğitim başlıyor...")
    global_step = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        running_haversine = 0.0
       
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch") as pbar:
            for batch_idx, (images, targets) in enumerate(train_loader):
                images = images.to(device)
                targets = targets.to(device)
               
                optimizer.zero_grad()
                outputs = model(images)  # [batch, 2]
               
                # 6.1 Loss (MSE)
                loss = criterion(outputs, targets)
               
                # 6.2 Geri yayılım
                loss.backward()
                optimizer.step()
               
                # 6.3 Coğrafi mesafe (Haversine) hesaplamak için:
                #    Metrik -> sadece raporlama amaçlı. Kaybı etkilemez.
                haversine_km = batch_haversine_metric(outputs.detach(), targets.detach())
               
                running_loss += loss.item()
                running_haversine += haversine_km
               
                # 6.4 İlerleme çubuğu
                pbar.set_postfix({
                    "loss": f"{running_loss/(batch_idx+1):.4f}",
                    "hav_km": f"{running_haversine/(batch_idx+1):.4f}"
                })
                pbar.update(1)
               
            # Her epoch sonu ortalamalar
            epoch_loss = running_loss / len(train_loader)
            epoch_hav  = running_haversine / len(train_loader)
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {epoch_loss:.4f} | HavDist (km): {epoch_hav:.4f}")
       
        # 6.5 Scheduler Güncellemesi
        # İlk WARMUP_EPOCHS sabit, sonrasında Cosine
        if epoch >= WARMUP_EPOCHS:
            scheduler.step()
   
    print("Eğitim tamamlandı!")

    # (Opsiyonel) Model kaydetmek
    # torch.save(model.state_dict(), "/kaggle/working/zscore_vit_mlp_adamw.pth")


##############################################
# EĞİTİM KODU (Z-SCORE + ViT MODEL)
##############################################

import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transformers import ViTModel
from tqdm import tqdm

# ===========================================
# 1. KULLANICI PARAMETRELERİ ve Z-SCORE DEĞERLERİ
# ===========================================
CSV_FILE  = "/kaggle/input/datathon-ai-24/train.csv"
IMG_DIR   = "/kaggle/input/datathon-ai-24/train"

# Bu değerleri, veri setinizi istatistiksel olarak analiz ederek bulduğunuzu varsayıyoruz
LAT_MEAN = 41.105763
LAT_STD  = 0.002358
LON_MEAN = 29.025191
LON_STD  = 0.004205

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4  # ViT için biraz daha düşük LR deneyebilirsiniz

# ==================================
# 2. Z-SCORE FONKSİYONLARI
# ==================================
def zscore_lat_lon(lat, lon):
    """
    (lat - LAT_MEAN) / LAT_STD, (lon - LON_MEAN) / LON_STD
    """
    lat_norm = (lat - LAT_MEAN) / LAT_STD
    lon_norm = (lon - LON_MEAN) / LON_STD
    return lat_norm, lon_norm

# Z-Score ters dönüşüm
def inverse_zscore_lat_lon(lat_norm, lon_norm):
    lat_real = lat_norm * LAT_STD + LAT_MEAN
    lon_real = lon_norm * LON_STD + LON_MEAN
    return lat_real, lon_real


# ==================================
# 3. CSV OKUMA
# ==================================
print("CSV dosyası okunuyor...")
df = pd.read_csv(CSV_FILE, sep=';')  # 'filename', 'latitude', 'longitude' sütunları olmalı.

# ==================================
# 4. DATASET TANIMI
# ==================================
class CampusDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, use_zscore=True):
        """
        :param use_zscore: True ise z-score normalizasyonu uygular
        """
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.use_zscore = use_zscore

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
       
        image_path = os.path.join(self.img_dir, row["filename"])
        lat = float(row["latitude"])
        lon = float(row["longitude"])
       
        # Resmi aç
        image = Image.open(image_path).convert("RGB")
       
        # Transform
        if self.transform:
            image = self.transform(image)
       
        # Z-Score normalizasyon
        if self.use_zscore:
            lat, lon = zscore_lat_lon(lat, lon)
       
        target = torch.tensor([lat, lon], dtype=torch.float32)
        return image, target

# ==================================
# 5. VERİ DÖNÜŞÜMLERİ (TRANSFORMS)
# ==================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Hugging Face ViT'nin genelde ImageNet normalizasyonuyla da iyi çalışır
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

# ==================================
# 6. DATALOADER
# ==================================
print("Dataset oluşturuluyor...")
train_dataset = CampusDataset(
    dataframe=df,
    img_dir=IMG_DIR,
    transform=train_transform,
    use_zscore=True  # Z-Score uygulamak için True
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# ==================================
# 7. MODEL TANIMI (ViT)
# ==================================
class ViTRegressor(nn.Module):
    def __init__(self, pretrained_model_name="google/vit-base-patch16-224-in21k", output_dim=2):
        super().__init__()
        # 1) Pretrained ViT gövdesi
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        # 2) CLS token boyutu -> 2 (lat, lon)
        self.fc = nn.Linear(self.vit.config.hidden_size, output_dim)

    def forward(self, x):
        # x -> [batch_size, 3, 224, 224]
        outputs = self.vit(x)
        # outputs.last_hidden_state -> [batch_size, seq_len, hidden_size]
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        preds = self.fc(cls_embedding)  # [batch_size, 2]
        return preds

print("Model oluşturuluyor...")
model = ViTRegressor("google/vit-base-patch16-224-in21k", output_dim=2)

# ==================================
# 8. GPU AYARI
# ==================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if torch.cuda.device_count() > 1:
    print(f"{torch.cuda.device_count()} GPU bulundu! DataParallel kullanılıyor.")
    model = nn.DataParallel(model)

# ==================================
# 9. KAYIP FONK. ve OPTİMİZER
# ==================================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==================================
# 10. EĞİTİM DÖNGÜSÜ
# ==================================
print("Eğitim başlıyor...")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
   
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch") as pbar:
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
           
            optimizer.zero_grad()
            outputs = model(images)  # [batch_size, 2]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
           
            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{running_loss/(batch_idx+1):.4f}"})
            pbar.update(1)
   
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Ortalama Kayıp: {epoch_loss:.4f}")

print("Eğitim tamamlandı!")

# (Opsiyonel) Modeli kaydetmek isterseniz:
torch.save(model.state_dict(), "/kaggle/working/zscore_vit.pth")
