FROM python:3.11

# Install dependencies
RUN pip install --no-cache-dir hic-straw numpy matplotlib tqdm torch torchvision torchaudio