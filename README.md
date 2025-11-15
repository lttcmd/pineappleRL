<b> 1. Go to your project folder: </b> <br>
cd C:\path\to\project

2. Create a virtual environment: <br>
python -m venv .venv

3. Activate the virtual environment: <br>
.venv\Scripts\Activate

4. Install PyTorch with CUDA 12.1: <br>
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

5. Check that the GPU is available: <br>
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

6. Install the remaining required packages: <br>
pip install numpy
pip install tqdm
pip install matplotlib
pip install tensorboard

7. Start training: <br>
python train.py
