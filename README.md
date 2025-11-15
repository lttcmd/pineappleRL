1. Go to your project folder: <br>
cd C:\path\to\project

2. Create a virtual environment: <br>
python -m venv .venv

3. Activate the virtual environment:
.venv\Scripts\Activate

4. Install PyTorch with CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

5. Check that the GPU is available:
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

6. Install the remaining required packages:
pip install numpy
pip install tqdm
pip install matplotlib
pip install tensorboard

7. Start training:
python train.py
