# Gemini Project Context: DGMR

## Project Overview
This project is an implementation of DeepMind's **Deep Generative Model of Radar (DGMR)**, a Skillful Nowcasting GAN for precipitation forecasting, as described in the Nature paper "[Skillful Precipitation Nowcasting using Deep Generative Models of Radar](https://arxiv.org/abs/2104.00954)".

- **Main Technologies:** Python, PyTorch, PyTorch Lightning, HuggingFace Hub/Datasets, Weights & Biases (W&B).
- **Architecture:** 
  - **Generator:** Composed of a `ContextConditioningStack` (extracts features from input radar frames), a `LatentConditioningStack` (processes latent noise), and a `Sampler` (uses ConvGRU layers to generate future frames).
  - **Discriminator:** Includes both `SpatialDiscriminator` (ensures spatial consistency) and `TemporalDiscriminator` (ensures temporal consistency across frames).
  - **Model Wrapper:** The `DGMR` class (in `dgmr/dgmr.py`) inherits from `pl.LightningModule` and manages the complex training loop, including manual optimization for GAN training.
- **Data:** Uses the UK Nimrod radar dataset (available via HuggingFace Datasets) and supports MRMS US radar data.

## Building and Running

### Installation
```bash
# Clone the repository and install dependencies
pip install -r requirements.txt
pip install -e .
```

### Training
The training script is located in `train/run.py`. It uses PyTorch Lightning and W&B for logging.
```bash
python train/run.py
```
*Note: Training requires access to HuggingFace datasets and a W&B account.*

### Testing
The project uses `pytest` for unit and integration tests.
```bash
pytest tests/
```

### Pretrained Weights
Pretrained weights can be loaded directly from HuggingFace Hub:
```python
from dgmr import DGMR
model = DGMR.from_pretrained("openclimatefix/dgmr")
```

## Development Conventions

- **Code Style:** 
  - Follows **Google-style docstrings**.
  - Linting and formatting are managed by `ruff` and `black`.
  - Line length is set to **100 characters**.
- **PyTorch Lightning Usage:**
  - The `DGMR` model uses **manual optimization** (`self.automatic_optimization = False`) to handle the alternating generator and discriminator updates characteristic of GANs.
  - Checkpointing is performed using `torch.utils.checkpoint` to save memory during training.
- **Hooks:** Pre-commit hooks are used for linting (`ruff`), formatting (`black`, `prettier`), and basic checks (trailing whitespace, private keys).
- **Architecture Patterns:**
  - Modular components: Stacks, Sampler, and Discriminators are implemented as standalone PyTorch modules.
  - Use of `einops` for tensor manipulations.
  - Support for `PyTorchModelHubMixin` to easily push/pull models to/from HuggingFace.
