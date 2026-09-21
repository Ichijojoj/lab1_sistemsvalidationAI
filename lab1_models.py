
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lab1_core import (
    ACTIVATIONS,
    BATCH_SIZE,
    EPOCHS,
    HIDDEN_DIM,
    LEARNING_RATE,
    MODEL_DIR,
    N_CLASSES,
    N_PIXELS,
    SEED,
    softmax_np,
)

# ===========================================================================
# PyTorch
# ===========================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


def torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_torch_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision("high")


class TwoLayerPerceptron(nn.Module):
    """Двухслойный перцептрон: Linear(784, H) -> Act -> Dropout -> Linear(H, 10).

    Выход — логиты. Softmax в модель не включён.
    """

    def __init__(self, hidden: int = HIDDEN_DIM, activation: str = "relu", dropout_p: float = 0.2):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(activation)
        self.activation_name = activation
        self.dropout_p = float(dropout_p)
        self.fc1 = nn.Linear(N_PIXELS, hidden)
        self.fc2 = nn.Linear(hidden, N_CLASSES)
        self.dropout = nn.Dropout(p=self.dropout_p)
        self._act = {"relu": F.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}[activation]
        self._init_xavier()

    def _init_xavier(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._act(self.fc1(x))
        h = self.dropout(h)
        return self.fc2(h)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def first_layer_weights(self) -> np.ndarray:
        return self.fc1.weight.detach().cpu().numpy()


def _gpu_tensors(x: np.ndarray, y: np.ndarray, device: torch.device):
    xt = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=device)
    yt = torch.as_tensor(np.ascontiguousarray(y), dtype=torch.int64, device=device)
    return xt, yt


@torch.no_grad()
def _eval_tensors(model: nn.Module, x: torch.Tensor, y: torch.Tensor):
    model.eval()
    logits = model(x)
    loss = float(F.cross_entropy(logits, y, reduction="mean"))
    acc = float((logits.argmax(dim=1) == y).float().mean())
    return loss, acc


def train_torch(
    x_train,
    y_train,
    x_val,
    y_val,
    activation: str,
    dropout_p: float,
    hidden: int = HIDDEN_DIM,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    seed: int = SEED,
    patience: int = 4,
) -> dict:
    set_torch_seed(seed)
    device = torch_device()
    model = TwoLayerPerceptron(hidden=hidden, activation=activation, dropout_p=dropout_p).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # CrossEntropyLoss = LogSoftmax + NLLLoss: softmax внутри функции потерь
    crit = nn.CrossEntropyLoss()

    x_tr, y_tr = _gpu_tensors(x_train, y_train, device)
    x_va, y_va = _gpu_tensors(x_val, y_val, device)
    n = int(x_tr.shape[0])

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_state, best_val, wait = None, float("inf"), 0
    t0 = time.perf_counter()

    for _epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x_tr.index_select(0, idx)), y_tr.index_select(0, idx))
            loss.backward()
            opt.step()
        tr_loss, tr_acc = _eval_tensors(model, x_tr, y_tr)
        va_loss, va_acc = _eval_tensors(model, x_va, y_va)
        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        hist["train_acc"].append(tr_acc)
        hist["val_acc"].append(va_acc)
        if va_loss < best_val - 1e-4:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "model": model,
        "history": hist,
        "device": str(device),
        "n_params": model.n_parameters(),
        "train_time_sec": elapsed,
        "epochs_run": len(hist["train_loss"]),
        "best_val_loss": best_val,
        "framework": "pytorch",
        "activation": activation,
        "dropout_p": dropout_p,
        "hidden": hidden,
        "loss": "CrossEntropyLoss (softmax внутри)",
        "optimizer": f"Adam(lr={lr})",
        "softmax": "внутри CrossEntropyLoss; явно при выводе вероятностей",
    }


@torch.no_grad()
def predict_torch(model: nn.Module, x: np.ndarray, device=None) -> tuple[np.ndarray, np.ndarray]:
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    x_t = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=device)
    logits = model(x_t).detach().cpu().numpy()
    proba = softmax_np(logits)
    pred = proba.argmax(axis=1)
    return pred, proba


def torch_predict_fn(model: nn.Module):
    device = next(model.parameters()).device

    def _fn(xb: np.ndarray) -> np.ndarray:
        _, proba = predict_torch(model, xb, device=device)
        return proba

    return _fn


def torch_saliency(model: nn.Module, image: np.ndarray, target_class: int | None = None) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    x = torch.tensor(image.reshape(1, -1), dtype=torch.float32, device=device, requires_grad=True)
    logits = model(x)
    cls = int(logits.argmax(dim=1).item()) if target_class is None else int(target_class)
    logits[0, cls].backward()
    sal = x.grad.detach().abs().cpu().numpy().reshape(28, 28)
    return sal


def save_torch(model: nn.Module, path: Path) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "activation": model.activation_name,
            "dropout_p": model.dropout_p,
            "hidden": model.fc1.out_features,
        },
        path,
    )


def load_torch(path: Path, device=None) -> TwoLayerPerceptron:
    device = device or torch_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = TwoLayerPerceptron(
        hidden=ckpt["hidden"], activation=ckpt["activation"], dropout_p=ckpt["dropout_p"]
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device)

# JAX
import jax
import jax.numpy as jnp
import optax


def set_jax_seed(seed: int = SEED):
    return jax.random.PRNGKey(seed)


def _xavier_uniform(key, fan_in, fan_out):
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, (fan_in, fan_out), minval=-limit, maxval=limit)


def init_jax_params(key, hidden: int = HIDDEN_DIM):
    k1, k2 = jax.random.split(key)
    return {
        "W1": _xavier_uniform(k1, N_PIXELS, hidden),
        "b1": jnp.zeros((hidden,)),
        "W2": _xavier_uniform(k2, hidden, N_CLASSES),
        "b2": jnp.zeros((N_CLASSES,)),
    }


def jax_n_params(params) -> int:
    return int(sum(p.size for p in jax.tree_util.tree_leaves(params)))


def _activate(name: str, h):
    if name == "relu":
        return jax.nn.relu(h)
    if name == "tanh":
        return jnp.tanh(h)
    if name == "sigmoid":
        return jax.nn.sigmoid(h)
    raise ValueError(name)


def jax_forward(params, x, activation: str, dropout_p: float, train: bool, key=None):
    h = x @ params["W1"] + params["b1"]
    h = _activate(activation, h)
    if train and dropout_p > 0:
        keep = jax.random.bernoulli(key, 1.0 - dropout_p, h.shape)
        h = h * keep / (1.0 - dropout_p)
    logits = h @ params["W2"] + params["b2"]
    return logits


def _jax_loss(params, x, y, activation, dropout_p, train, key):
    logits = jax_forward(params, x, activation, dropout_p, train, key)
    # softmax внутри optax.softmax_cross_entropy_with_integer_labels
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    acc = (logits.argmax(axis=-1) == y).mean()
    return loss, acc


def train_jax(
    x_train,
    y_train,
    x_val,
    y_val,
    activation: str,
    dropout_p: float,
    hidden: int = HIDDEN_DIM,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    seed: int = SEED,
    patience: int = 4,
) -> dict:
    key = set_jax_seed(seed)
    key, k_init = jax.random.split(key)
    params = init_jax_params(k_init, hidden=hidden)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    x_tr = jnp.asarray(x_train, dtype=jnp.float32)
    y_tr = jnp.asarray(y_train, dtype=jnp.int32)
    x_va = jnp.asarray(x_val, dtype=jnp.float32)
    y_va = jnp.asarray(y_val, dtype=jnp.int32)
    n = int(x_tr.shape[0])

    @jax.jit
    def step(params, opt_state, xb, yb, drop_key):
        def loss_fn(p):
            loss, _ = _jax_loss(p, xb, yb, activation, dropout_p, True, drop_key)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def eval_full(params, x, y):
        loss, acc = _jax_loss(params, x, y, activation, 0.0, False, None)
        return loss, acc

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_params, best_val, wait = params, float("inf"), 0
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        key, k_perm, k_drop = jax.random.split(key, 3)
        perm = jax.random.permutation(k_perm, n)
        xp, yp = x_tr[perm], y_tr[perm]
        for start in range(0, n, BATCH_SIZE):
            xb = xp[start : start + BATCH_SIZE]
            yb = yp[start : start + BATCH_SIZE]
            k_drop, kd = jax.random.split(k_drop)
            params, opt_state, _ = step(params, opt_state, xb, yb, kd)

        tr_loss, tr_acc = eval_full(params, x_tr, y_tr)
        va_loss, va_acc = eval_full(params, x_va, y_va)
        tr_loss, tr_acc = float(tr_loss), float(tr_acc)
        va_loss, va_acc = float(va_loss), float(va_acc)
        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        hist["train_acc"].append(tr_acc)
        hist["val_acc"].append(va_acc)
        if va_loss < best_val - 1e-4:
            best_val = va_loss
            best_params = params
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    elapsed = time.perf_counter() - t0
    # JAX считает train loss/acc по всему train без dropout, сопоставимо с _eval_torch
    return {
        "params": best_params,
        "history": hist,
        "device": str(jax.default_backend()),
        "n_params": jax_n_params(best_params),
        "train_time_sec": elapsed,
        "epochs_run": len(hist["train_loss"]),
        "best_val_loss": best_val,
        "framework": "jax",
        "activation": activation,
        "dropout_p": dropout_p,
        "hidden": hidden,
        "loss": "optax.softmax_cross_entropy_with_integer_labels (softmax внутри)",
        "optimizer": f"Adam(lr={lr})",
        "softmax": "внутри softmax_cross_entropy; явно при выводе вероятностей",
    }


def predict_jax(params, x: np.ndarray, activation: str) -> tuple[np.ndarray, np.ndarray]:
    xt = jnp.asarray(np.ascontiguousarray(x), dtype=jnp.float32)
    logits = np.asarray(jax_forward(params, xt, activation, 0.0, False, None))
    proba = softmax_np(logits)
    return proba.argmax(axis=1), proba


def jax_predict_fn(params, activation: str):
    def _fn(xb: np.ndarray) -> np.ndarray:
        _, proba = predict_jax(params, xb, activation)
        return proba

    return _fn


def jax_first_layer_weights(params) -> np.ndarray:
    # W1: (784, hidden) -> (hidden, 784) как в PyTorch
    return np.asarray(params["W1"]).T


def jax_saliency(params, image: np.ndarray, activation: str, target_class: int | None = None) -> np.ndarray:
    x = jnp.asarray(image.reshape(1, -1), dtype=jnp.float32)

    def score(inp):
        logits = jax_forward(params, inp, activation, 0.0, False, None)
        cls = logits.argmax(axis=-1)[0] if target_class is None else target_class
        return logits[0, cls]

    g = jax.grad(score)(x)
    return np.abs(np.asarray(g)).reshape(28, 28)


def save_jax(bundle: dict, path: Path) -> None:
    np.savez(
        path,
        W1=np.asarray(bundle["params"]["W1"]),
        b1=np.asarray(bundle["params"]["b1"]),
        W2=np.asarray(bundle["params"]["W2"]),
        b2=np.asarray(bundle["params"]["b2"]),
        activation=np.array(bundle["activation"]),
        dropout_p=np.array(bundle["dropout_p"]),
        hidden=np.array(bundle["hidden"]),
    )


def load_jax(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    params = {
        "W1": jnp.asarray(z["W1"]),
        "b1": jnp.asarray(z["b1"]),
        "W2": jnp.asarray(z["W2"]),
        "b2": jnp.asarray(z["b2"]),
    }
    return {
        "params": params,
        "activation": str(z["activation"]),
        "dropout_p": float(z["dropout_p"]),
        "hidden": int(z["hidden"]),
    }
