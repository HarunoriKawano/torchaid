# torchaid

**torchaid** is a PyTorch utility library that provides structured abstractions and reusable components to streamline the deep learning training pipeline.

## Features

- **Structured training abstractions** — base classes for metrics and settings built on Pydantic v2
- **Training framework** — a full training loop with mixed-precision support, automatic checkpointing, metric logging (CSV), and early stopping
- **Transformer modules** — standard and relative-position-aware Transformer encoder layers with multi-head self-attention
- **Task templates** — ready-to-use implementation for multi-label classification
- **Utilities** — dataset splitting, random seed management, attention mask generation, and JSON-to-Pydantic loading
- **Learning rate schedulers** — cosine decay with linear warm-up, and triangular2 cyclic scheduling

## Requirements

- Python 3.10+
- PyTorch 2.0+

## Installation

```bash
pip install torchaid
```

Or install from source:

```bash
git clone https://github.com/harunori-kawano/torchaid.git
cd torchaid
pip install -e .
```

## Quick Start

### 1. Implement your model

Batches are plain `dict[str, Any]`. `forward` returns a `(outputs, error)` tuple — set `error` to a non-`None` value to signal a recoverable per-batch error; the framework will skip backpropagation and log it to stderr.

```python
from torchaid import TaskModule, Mode
from typing import Any, Optional
from torch import nn

class MyModel(TaskModule):
    def __init__(self, vocab_size: int, num_classes: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 128)
        self.classifier = nn.Linear(128, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, mode: Mode, batch: dict[str, Any]) -> tuple[dict[str, Any], Optional[Any]]:
        x = self.embed(batch["input_ids"]).mean(dim=1)
        logits = self.classifier(x)
        loss = self.criterion(logits, batch["labels"])
        if mode == Mode.TRAIN:
            return {"loss": loss}, None
        return {"loss": loss, "logits": logits}, None
```

### 2. Define metrics and settings

```python
from torchaid import BaseMetrics, BaseSettings, BaseMetricCalculator
from typing import Optional, Any

class MyMetrics(BaseMetrics):
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None

class MySettings(BaseSettings):
    batch_size: int = 32
    max_epoch_num: int = 10

class MyCalculator(BaseMetricCalculator[MyMetrics]):
    def __init__(self):
        super().__init__(MyMetrics())
        self._losses: list[float] = []

    def train_step(self, outputs: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
        loss = outputs["loss"].item()
        self._losses.append(loss)
        return {"loss": loss}

    def val_step(self, outputs: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
        return self.train_step(outputs, batch)

    def test_step(self, outputs: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
        return self.train_step(outputs, batch)

    def check(self) -> bool:
        import statistics
        self.metrics.train_loss = statistics.mean(self._losses)
        return True

    def test(self): pass

    def reset(self):
        self._losses.clear()
```

### 3. Train

```python
import torch
from torchaid.core.trainer import TrainFramework

settings = MySettings(batch_size=32, max_epoch_num=10, device="cuda")
model = MyModel(vocab_size=1000, num_classes=5)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

framework = TrainFramework(
    model=model,
    ls=settings,
    metric_calculator=MyCalculator(),
    optimizer=optimizer,
)

framework.train(train_dataset, val_dataset, save_dir="./outputs")
```

## Module Overview

| Module | Description |
|--------|-------------|
| `torchaid.core` | Base classes (`BaseMetrics`, `BaseSettings`, `TaskModule`, `BaseMetricCalculator`, `Mode`) and `TrainFramework` |
| `torchaid.templates.multilabel_classification` | Complete template for multi-label classification |
| `torchaid.extras.modules.transformer` | `Transformer`, `TransformerWithRelativePosition`, and sub-modules |
| `torchaid.extras.modules.positional_encoders` | `PositionalEmbedding`, `RelativePositionEmbedding` |
| `torchaid.extras.utils` | `split_dataset`, `set_random_seed`, `make_attention_mask`, `json_to_instance` |
| `torchaid.extras.scheduler` | `get_cosine_scheduler`, `get_cycle_scheduler` |

## Template: Multi-Label Classification

```python
from torchaid.templates import multilabel_classification as mlc
from torchaid.core.trainer import TrainFramework
from torch import nn
import torch

backbone = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
model = mlc.MultiLabelClassification(backbone)
optimizer = torch.optim.Adam(model.parameters())

framework = TrainFramework(
    model=model,
    ls=settings,
    metric_calculator=mlc.MetricsCalculator(),
    optimizer=optimizer,
)
```

## Extras

### Cosine Decay Scheduler

```python
from torchaid.extras.scheduler import get_cosine_scheduler

scheduler = get_cosine_scheduler(
    optimizer, warmup_steps=500, max_steps=10000
)
```

### Dataset Split

```python
from torchaid.extras.utils import split_dataset

train, val, test = split_dataset(dataset, ratios=[8, 1, 1], seed=42)
```

### Relative Position Transformer

```python
from torchaid.extras.modules.transformer import TransformerWithRelativePosition

layer = TransformerWithRelativePosition(
    hidden_size=256,
    intermediate_size=1024,
    num_attention_heads=8,
    dropout_probability=0.1,
    max_length=512,
    with_cls=True,
)
```

## License

MIT License. See [LICENSE](LICENSE) for details.
