import json
import matplotlib.pyplot as plt
import os

trainer_state_path = "./phi-3-deepseek-finetuned/checkpoint-490/trainer_state.json"

with open(trainer_state_path, "r") as f:
    trainer_state = json.load(f)

log_history = trainer_state["log_history"]

epochs = []
losses = []
learning_rates = []
grad_norms = []

for entry in log_history:
    if "epoch" in entry:
        epochs.append(entry["epoch"])
        losses.append(entry.get("loss", None))
        learning_rates.append(entry.get("learning_rate", None))
        grad_norms.append(entry.get("grad_norm", None))

log_dir = "./logs/log_temp"
os.makedirs(log_dir, exist_ok=True)

line_width = 1
marker_size = 3

plt.figure(figsize=(6, 4))
plt.plot(epochs, losses, marker="o", linestyle="-", linewidth=line_width, markersize=marker_size, color="b", label="Loss")
plt.title("Loss vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig(os.path.join(log_dir, "loss_curve.png"))
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(epochs, learning_rates, marker="s", linestyle="--", linewidth=line_width, markersize=marker_size, color="g", label="Learning Rate")
plt.title("Learning Rate vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.legend()
plt.grid()
plt.savefig(os.path.join(log_dir, "learning_rate_curve.png"))
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(epochs, grad_norms, marker="^", linestyle=":", linewidth=line_width, markersize=marker_size, color="r", label="Grad Norm")
plt.title("Gradient Norm vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Grad Norm")
plt.legend()
plt.grid()
plt.savefig(os.path.join(log_dir, "grad_norm_curve.png"))
plt.close()


print(f"所有图像已保存到 {log_dir}")
