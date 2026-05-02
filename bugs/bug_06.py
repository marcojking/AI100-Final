"""Bug 06: Learning rate = 10.0, loss explodes to NaN"""
import pandas as pd, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings; warnings.filterwarnings('ignore')

df = pd.read_csv('../dataset.csv', encoding='utf-8')
genre_map = {
    'Rock': ['rock','alt-rock','alternative','hard-rock','punk','punk-rock','grunge','psych-rock','rock-n-roll','rockabilly','emo','goth','indie','ska'],
    'Metal': ['metal','heavy-metal','death-metal','black-metal','metalcore','grindcore','hardcore','industrial'],
    'Pop': ['pop','indie-pop','synth-pop','k-pop','j-pop','cantopop','mandopop','j-idol','pop-film','power-pop','british','swedish'],
    'Electronic': ['edm','electro','electronic','house','chicago-house','deep-house','detroit-techno','techno','minimal-techno','progressive-house','trance','hardstyle','drum-and-bass','dubstep','breakbeat','garage','idm','trip-hop','dub'],
    'Hip-Hop/R&B': ['hip-hop','r-n-b','reggaeton','dancehall'],
    'Jazz/Blues': ['jazz','blues','soul','funk','groove'],
    'Classical': ['classical','piano','opera','guitar','new-age','ambient','sleep','study'],
    'Folk/Country': ['folk','country','acoustic','bluegrass','honky-tonk','singer-songwriter','songwriter'],
    'Latin/World': ['latin','latino','salsa','samba','forro','sertanejo','pagode','mpb','brazil','afrobeat','indian','iranian','turkish','malay','tango','reggae','french','german','spanish','world-music'],
    'Dance/Other': ['dance','club','disco','party','happy','chill','romance','sad','anime','disney','children','kids','comedy','show-tunes','j-dance','j-rock','gospel'],
}
reverse_map = {g: sg for sg, gs in genre_map.items() for g in gs}
df['super_genre'] = df['track_genre'].map(reverse_map)
df = df.dropna(subset=['super_genre'])
audio_features = ['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','time_signature','duration_ms']
X = df[audio_features].values
le = LabelEncoder()
y = le.fit_transform(df['super_genre'].values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.LongTensor(y_train)
X_test_t  = torch.FloatTensor(X_test_scaled)
y_test_t  = torch.LongTensor(y_test)
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)

class GenreClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),        nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),         nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.network(x)

model = GenreClassifier(13, 10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=10.0, weight_decay=1e-4)  # BUG: lr=10.0

print("=== BUG 06: Learning rate = 10.0 ===")
EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_test_t), dim=1)
        acc = (preds == y_test_t).float().mean().item()
    print(f"Epoch {epoch+1}/{EPOCHS}  loss={avg_loss:.4f}  test_acc={acc:.4f}")
    if str(avg_loss) == 'nan':
        print("Loss is NaN, training has diverged.")
        break
