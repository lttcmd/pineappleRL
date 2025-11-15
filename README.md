
#1. For Windows (CUDA 12.1):
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

  For Linux (CUDA 12.1):
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

# 2. Verify GPU is detected
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# 3. Install other dependencies
pip install -r requirements.txt

# 4. Test the installation
python test_one_hand.py

# 5. Start training
python train.py
```
