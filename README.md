# mnist-mlp

Bài tập MNIST MLP (PyTorch), môn Machine Learning.

## Cách chạy

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python mnist_mlp.py
```

## Kết quả thí nghiệm

BATCH = 128 cho tất cả các lần chạy.

| EPOCHS | HIDDEN | LR | Test acc |
|---|---|---|---|
| 3 | 128 | 1e-3 | 0.9606 |
| 5 | 256 | 1e-3 | 0.9710 |
| 5 | 256 | 1e-2 | 0.9672 |
| 5 | 256 | 1e-4 | 0.9195 |

## Nhận xét

- LR = 1e-3 cho kết quả tốt nhất (test acc 0.9710).
- LR = 1e-2 học nhanh lúc đầu nhưng val loss tăng từ epoch 4, kết quả kém hơn.
- LR = 1e-4 học quá chậm, sau 5 epoch loss vẫn đang giảm và chưa hội tụ.