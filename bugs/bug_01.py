"""Bug 01: fit_transform on test set instead of transform (data leakage / wrong normalization)"""
import sys
sys.path.insert(0, '..')
from bugs.base import *

print("=== BUG 01: fit_transform on test set ===")
scaler2 = StandardScaler()
X_train_scaled2 = scaler2.fit_transform(X_train)
X_test_scaled2  = scaler2.fit_transform(X_test)   # BUG: fit_transform instead of transform

X_train_t2 = torch.FloatTensor(X_train_scaled2)
y_train_t2 = torch.LongTensor(y_train)
X_test_t2  = torch.FloatTensor(X_test_scaled2)
y_test_t2  = torch.LongTensor(y_test)
train_loader2 = DataLoader(TensorDataset(X_train_t2, y_train_t2), batch_size=256, shuffle=True)

model2 = GenreClassifier(input_size=13, num_classes=10)
criterion2 = nn.CrossEntropyLoss()
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001, weight_decay=1e-4)

for epoch in range(EPOCHS):
    model2.train()
    total_loss = 0
    for X_batch, y_batch in train_loader2:
        optimizer2.zero_grad()
        loss = criterion2(model2(X_batch), y_batch)
        loss.backward()
        optimizer2.step()
        total_loss += loss.item()
    model2.eval()
    with torch.no_grad():
        preds = torch.argmax(model2(X_test_t2), dim=1)
        acc = (preds == y_test_t2).float().mean().item()
    print(f"Epoch {epoch+1}/{EPOCHS}  loss={total_loss/len(train_loader2):.4f}  test_acc={acc:.4f}")

print("\nNote: no crash, but test accuracy differs from base because test features")
print("are normalized with different mean/std than training features.")
