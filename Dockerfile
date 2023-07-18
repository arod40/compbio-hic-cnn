FROM python:3.11

# Install dependencies
RUN pip install --no-cache-dir torch torchvision torchaudio
RUN pip install --no-cache-dir hic-straw numpy matplotlib fire scikit-learn tqdm pandas