# ⚡ Transformer from Scratch (PyTorch)

Minimal **Transformer NMT** model built from scratch. Translate **English → Hindi/Italian** with PyTorch. No Hugging Face!  

---

## Features
- Step-by-step Transformer: Embeddings, Positional Encoding, Multi-Head Attention  
- Word-level tokenization + masks  
- Checkpointing & TensorBoard logging  
- GPU-ready  

---

## Quickstart
```bash
git clone https://github.com/Sreys10/transformer-from-scratch.git
cd transformer-from-scratch
pip install torch datasets tokenizers tensorboard tqdm
python train.py
tensorboard --logdir=runs
