import json
import matplotlib.pyplot as plt
import os

# 读取 trainer_state.json
trainer_state_path = "./phi-3-deepseek-finetuned/checkpoint-7029/trainer_state.json"

with open(trainer_state_path, "r") as f:
    trainer_state = json.load(f)

# 提取 log_history（包含 loss、learning_rate 和 grad_norm）
log_history = trainer_state["log_history"]

# 初始化数据存储
epochs = []
losses = []
learning_rates = []
grad_norms = []

# 遍历 log_history 记录
for entry in log_history:
    if "epoch" in entry:  # 只取包含 epoch 记录的部分
        epochs.append(entry["epoch"])
        losses.append(entry.get("loss", None))
        learning_rates.append(entry.get("learning_rate", None))
        grad_norms.append(entry.get("grad_norm", None))

# 创建日志目录
log_dir = "./log1"
os.makedirs(log_dir, exist_ok=True)

# 绘制并保存 Loss 曲线
plt.figure(figsize=(6, 4))
plt.plot(epochs, losses, marker="o", linestyle="-", color="b", label="Loss")
plt.title("Loss vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig(os.path.join(log_dir, "loss_curve.png"))
plt.close()

# 绘制并保存 Learning Rate 曲线
plt.figure(figsize=(6, 4))
plt.plot(epochs, learning_rates, marker="s", linestyle="--", color="g", label="Learning Rate")
plt.title("Learning Rate vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.legend()
plt.grid()
plt.savefig(os.path.join(log_dir, "learning_rate_curve.png"))
plt.close()

# 绘制并保存 Grad Norm 曲线
plt.figure(figsize=(6, 4))
plt.plot(epochs, grad_norms, marker="^", linestyle=":", color="r", label="Grad Norm")
plt.title("Gradient Norm vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Grad Norm")
plt.legend()
plt.grid()
plt.savefig(os.path.join(log_dir, "grad_norm_curve.png"))
plt.close()

print(f"所有图像已保存到 {log_dir}")
