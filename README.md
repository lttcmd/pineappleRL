<b> 1. pject folder: </b> <br>
cd C:\path\to\project

<b> 2. create environment: </b> <br>
python -m venv .venv

<b> 3. Activate environment: </b>
.venv\Scripts\Activate

<b> 4. Install PyTorch with CUDA 12.1: </b>
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

<b> 5. Check that the GPU is available: </b>
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

<b> 6. Install the remaining required packages: </b>
pip install numpy
pip install tqdm
pip install matplotlib
pip install tensorboard

<b> 7. Start training: </b>
python train.py

Training saves after 10,000 hands and prints results. 
Interested to see what number of hands it gets per second....
