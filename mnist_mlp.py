# mnist_mlp.py — 10주차 MNIST MLP를 VS Code에서 '파이썬 파일'로 실행하는 버전
# 실행: 터미널에서  python mnist_mlp.py   (가상환경 .venv가 켜진 상태에서)
import time, json
import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")                      # 창을 띄우지 않고 그림을 파일로만 저장 (터미널 실행용)
import matplotlib.pyplot as plt

EPOCHS, BATCH, LR, HIDDEN = 5, 128, 1e-3, 256  # ← 바꿔 보며 실험하는 값들 (디버깅 연습 포인트)

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch {torch.__version__} | device: {device}")

# 1. 데이터 (첫 실행 때 ./data 폴더에 자동 다운로드, 약 11 MB)
transform = transforms.ToTensor()
train_full = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_ds, val_ds = random_split(train_full, [50_000, 10_000], generator=torch.Generator().manual_seed(0))
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1000)
test_loader = DataLoader(test_ds, batch_size=1000)
print(f"train {len(train_ds)} / val {len(val_ds)} / test {len(test_ds)}")

# 2. 모델·손실·옵티마이저 (9주차 4조각)
model = nn.Sequential(nn.Flatten(), nn.Linear(784, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 10)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def evaluate(loader):
    model.eval()
    total, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total += loss_fn(logits, yb).item() * len(xb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            n += len(xb)
    return total / n, correct / n


# 3. 학습 루프
hist = {"train_loss": [], "val_loss": [], "val_acc": []}
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    running, n = 0.0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * len(xb)
        n += len(xb)
    train_loss = running / n
    val_loss, val_acc = evaluate(val_loader)
    hist["train_loss"].append(train_loss); hist["val_loss"].append(val_loss); hist["val_acc"].append(val_acc)
    print(f"epoch {epoch} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val acc {val_acc:.4f}")
print(f"학습 시간 {time.time() - t0:.0f}초")

# 4. 테스트 한 번, 저장
test_loss, test_acc = evaluate(test_loader)
print(f"테스트 정확도 {test_acc:.4f}")
torch.save(model.state_dict(), "mnist_mlp.pt")
with open("history.json", "w") as f:
    json.dump(hist, f, indent=1)

plt.figure(figsize=(6, 4))
plt.plot(range(1, EPOCHS + 1), hist["train_loss"], "o-", label="train loss")
plt.plot(range(1, EPOCHS + 1), hist["val_loss"], "s--", label="val loss")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.grid(alpha=0.3); plt.title("MNIST MLP")
plt.savefig("loss_curve.png", dpi=120, bbox_inches="tight")
print("저장: mnist_mlp.pt, history.json, loss_curve.png")
