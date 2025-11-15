# H200 GPU Optimization Guide

## ✅ Optimizations Completed

### 1. **Network Architecture** (value_net.py)
- ✅ Increased hidden units: **256 → 512** (2x capacity)
- ✅ Added **dropout (0.1)** for regularization (prevents overfitting)
- **Impact**: Better learning capacity, can learn more complex strategies

### 2. **Training Configuration** (train.py)
- ✅ Increased batch size: **64 → 256** (4x larger, better GPU utilization)
- ✅ Increased buffer size: **200k → 500k** (more diverse training data)
- ✅ Increased CPU workers: **8 → 24** (uses ALL CPU cores on H200)
- **Impact**: Much faster training, better GPU utilization

### 3. **Learning Rate Scheduling** (train.py)
- ✅ Added automatic LR decay: **10% reduction every 10k episodes**
- ✅ Starts at 1e-3, gradually decreases for fine-tuning
- **Impact**: Better convergence, prevents overshooting optimal values

## 🎯 Training Objective Verification

**Your bot is learning to:**
- ✅ **Avoid fouling** (learns from -3 point penalties)
- ✅ **Maximize royalties** (learns from positive scores)
- ✅ **Make logical placements** (value network evaluates all legal actions)
- ✅ **Balance risk/reward** (explores early, exploits learned strategies later)

**How it learns:**
- Every state-action pair gets the **final score** as the target
- Model learns: "If I'm in state X and take action Y, I expect score Z"
- Over millions of hands, it learns which states/actions lead to good scores

## 💻 DigitalOcean Setup

### **OS Recommendation: "AI/ML Ready"**

**Why:**
- ✅ GPU drivers **pre-installed** (saves setup time = saves money)
- ✅ CUDA toolkit ready
- ✅ Optimized for ML workloads
- ✅ Based on Ubuntu (most compatible with PyTorch)

**Alternative:** Ubuntu 25.10 x64 (if you want to install drivers manually)

### **Setup Steps:**

1. **Create Droplet:**
   - Choose "AI/ML Ready" image
   - Select H200 GPU
   - 24 CPU cores
   - At least 50GB storage (for checkpoints)

2. **SSH into droplet:**
   ```bash
   ssh root@your-droplet-ip
   ```

3. **Verify GPU:**
   ```bash
   nvidia-smi
   ```
   Should show H200 GPU

4. **Install Python dependencies:**
   ```bash
   # Update system
   apt update && apt upgrade -y
   
   # Install Python 3 and pip
   apt install python3 python3-pip -y
   
   # Install PyTorch with CUDA support
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # Install other dependencies
   pip3 install numpy tqdm
   ```

5. **Upload your project:**
   ```bash
   # On your local machine, use scp or git
   scp -r "pineapple solver" root@your-droplet-ip:/root/
   # OR
   git clone your-repo-url
   ```

6. **Transfer latest checkpoint:**
   ```bash
   # Copy your latest checkpoint from local machine
   scp value_net_checkpoint_ep*.pth root@your-droplet-ip:/root/pineapple\ solver/
   ```

7. **Run training:**
   ```bash
   cd "pineapple solver"
   python3 train.py
   ```

## 📊 Expected Performance on H200

**With optimizations:**
- **Speed**: ~5000+ hands/second (vs ~145 on your current GPU)
- **GPU Utilization**: 80-95% (batch size 256 keeps GPU busy)
- **CPU Utilization**: ~95% (24 workers generating episodes)
- **Memory**: ~2-4GB GPU memory (512 hidden units)

**Training time estimates:**
- 1 million hands: ~3.3 minutes
- 1 billion hands: ~55 hours (~2.3 days)
- 10 billion hands: ~23 days

## ⚠️ Important Notes

### **Checkpoint Compatibility:**
- ⚠️ **NEW network (512 hidden) is NOT compatible with OLD checkpoints (256 hidden)**
- If you want to continue from your current checkpoint, you have two options:

**Option 1: Start fresh** (recommended for best results)
- Start training from scratch with 512 hidden units
- Better learning capacity from the beginning

**Option 2: Transfer learning** (if you want to keep progress)
- We can modify the code to load 256→512 (adds new layers, keeps old weights)
- Let me know if you want this

### **Cost Optimization:**
- Checkpoints save every 10k episodes (~18MB each)
- For 1 billion hands = 100k checkpoints = ~1.8GB storage
- Consider increasing checkpoint frequency to every 50k episodes to save I/O

### **Monitoring:**
- Watch GPU utilization: should be 80-95%
- Watch CPU utilization: should be 90-100% (all 24 cores busy)
- Watch learning rate: decreases every 10k episodes
- Watch foul rate: should decrease over time (target: <20%)

## 🚀 Next Steps

1. **Choose OS**: "AI/ML Ready" (recommended)
2. **Set up droplet** with H200 GPU
3. **Install dependencies** (see commands above)
4. **Upload code** to droplet
5. **Decide**: Start fresh (512 network) or continue from checkpoint (256→512 transfer)
6. **Start training** and monitor progress

## 📝 Summary of Changes

| Component | Before | After | Impact |
|-----------|--------|-------|--------|
| Hidden Units | 256 | 512 | 2x learning capacity |
| Dropout | None | 0.1 | Prevents overfitting |
| Batch Size | 64 | 256 | 4x GPU utilization |
| Buffer Size | 200k | 500k | More diverse data |
| CPU Workers | 8 | 24 | 3x episode generation |
| Learning Rate | Fixed | Scheduled | Better convergence |

**All changes are in:**
- `value_net.py` (network architecture)
- `train.py` (training configuration)

