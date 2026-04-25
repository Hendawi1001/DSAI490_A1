# Medical MNIST Autoencoders Project

This project contains code to train Autoencoders (AE) and Variational Autoencoders (VAE) on the Medical MNIST dataset. The project has been refactored into a standard Deep Learning project structure for better maintainability and modularity.

## Project Structure

```text
├── data/
│   ├── raw/                 # Store raw datasets (e.g. medical-mnist.zip) here
│   └── processed/           # Extracted datasets are placed here
├── models/                  # Trained models (.keras) and metadata (.json) are saved here
├── notebooks/               # Original and exploratory Jupyter notebooks
├── src/
│   ├── __init__.py
│   ├── data_processing.py   # Dataset loading and tf.data.Dataset pipeline
│   ├── model.py             # AE and VAE model definitions
│   └── train.py             # Main training script
├── tests/
│   ├── test_data_processing.py
│   └── test_model.py
├── README.md
└── requirements.txt
```

## Setup

1. **Install Requirements:**
   Make sure you have a Python environment set up (e.g. conda or venv).
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Placement:**
   Place the `medical-mnist.zip` file into the `data/raw/` directory. If you already have the unzipped data, you can place the directories directly into `data/processed/medical-mnist/`.

## Running the Training Script

To start training, simply run:
```bash
python src/train.py
```

### Options
You can configure the training script with various arguments:
```bash
python src/train.py --epochs 10 --batch_size 32 --latent_dim 128 --beta 1.5
```
Use `python src/train.py --help` for all available options.

## Evaluation and Visualization

To evaluate trained models and generate visualizations (reconstructions, latent space PCA, generated samples, loss curves, and denoising), run the evaluation script:
```bash
python src/evaluate.py
```
This will automatically load your models and save `.png` plots into the `reports/figures/` directory.

## Testing

Run the unit tests using `unittest`:
```bash
python -m unittest discover tests
```

## Performance and Optimization
- This project leverages `tf.data.Dataset` mapping and prefetching (`tf.data.AUTOTUNE`) for optimized and memory-efficient data loading during training.
