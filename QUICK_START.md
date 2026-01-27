# 🚀 Quick Start Guide

Get started with the assignment in 5 minutes!

## Option 1: Google Colab (Recommended for Beginners)

### Step 1: Upload Notebook
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click `File → Upload notebook`
3. Upload `CV_assignment2_group_4.ipynb`

### Step 2: Set Runtime to GPU
1. Click `Runtime → Change runtime type`
2. Select `GPU` (T4 or better)
3. Click `Save`

### Step 3: Install Dependencies
Run this cell first:
```python
!pip install torch torchvision transformers diffusers accelerate peft
!pip install pytesseract easyocr Pillow opencv-python numpy pandas matplotlib seaborn tqdm
!apt-get install tesseract-ocr
```

### Step 4: Run All Cells
- Click `Runtime → Run all`
- Wait for training to complete (~30-60 minutes on T4 GPU)

### Step 5: Export for Submission
- After completion, click `File → Download → Download .ipynb`
- Convert to PDF: `File → Print → Save as PDF`

---

## Option 2: Local Setup (For Advanced Users)

### Prerequisites Check
```bash
# Check Python version (3.8+ required)
python --version

# Check if GPU is available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check Tesseract
tesseract --version
```

### Quick Install
```bash
# Navigate to project
cd /Users/sahup3/Projects/MTech/CV_assignment_2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install everything
pip install -r requirements.txt

# Install Tesseract (if not installed)
# macOS: brew install tesseract
# Ubuntu: sudo apt-get install tesseract-ocr
```

### Run Notebook
```bash
# Start Jupyter
jupyter notebook

# Open CV_assignment2_group_4.ipynb and run all cells
```

---

## Expected Runtime

| Environment | Time | Notes |
|------------|------|-------|
| Colab (T4 GPU) | 30-60 min | ✅ Recommended |
| Colab (V100 GPU) | 20-40 min | ⚡ Fastest |
| Local (RTX 3080) | 30-45 min | ✅ Good |
| Local (CPU) | 3-5 hours | ❌ Not recommended |

---

## Quick Troubleshooting

### "CUDA out of memory"
```python
# In config cell, change:
BATCH_SIZE = 1  # Already set to 1
NUM_EPOCHS = 30  # Reduce from 50
```

### "Tesseract not found"
```python
# Add this after imports:
import pytesseract
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Adjust path
```

### "Model download slow"
```python
# Use cached model (after first run):
# Models are automatically cached in ~/.cache/huggingface/
```

---

## Verify Setup (Run This First!)

```python
# Test cell - run this before starting
import torch
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

import pytesseract
print(f"✓ Tesseract: {pytesseract.get_tesseract_version()}")

import transformers, diffusers, peft
print(f"✓ Transformers: {transformers.__version__}")
print(f"✓ Diffusers: {diffusers.__version__}")
print(f"✓ PEFT: {peft.__version__}")

print("\n🎉 All dependencies are installed correctly!")
```

---

## Expected Output Structure

After running the notebook, you should have:

```
CV_assignment_2/
├── data/
│   └── train/
│       ├── image_0000.png
│       ├── image_0001.png
│       ├── ...
│       └── metadata.csv
├── models/
│   ├── checkpoint_epoch_10/
│   ├── checkpoint_epoch_20/
│   ├── checkpoint_epoch_30/
│   ├── checkpoint_epoch_40/
│   ├── checkpoint_epoch_50/
│   └── final_lora_model/
└── outputs/
    ├── training_curves.png
    ├── generated_images_with_ocr.png
    ├── before_after_comparison.png
    ├── performance_metrics.png
    ├── evaluation_results.csv
    └── evaluation_summary.json
```

---

## Performance Expectations

### Training
- **Loss**: Should decrease from ~0.5 to ~0.1-0.2
- **Convergence**: Visible after 20-30 epochs
- **Model Size**: ~50MB (LoRA weights only)

### Evaluation
- **Character Accuracy**: 60-85%
- **Exact Match Rate**: 40-70%
- **Best for**: Short, simple text (e.g., "STOP", "EXIT")
- **Challenging**: Long text, complex fonts

---

## Customization Tips

### Generate More Training Data
```python
# In section 3.2, change:
train_df = create_training_dataset(num_samples=200)  # Instead of 100
```

### Add Custom Text
```python
# Add your own text to SAMPLE_TEXTS list:
SAMPLE_TEXTS = [
    "YOUR TEXT",
    "ANOTHER TEXT",
    # ... existing texts
]
```

### Faster Training (Lower Quality)
```python
# In Config class:
NUM_EPOCHS = 20  # Reduce from 50
LEARNING_RATE = 2e-4  # Increase for faster convergence
```

### Better Quality (Slower Training)
```python
# In Config class:
NUM_EPOCHS = 100  # Increase from 50
LORA_RANK = 8  # Increase from 4
```

---

## Submission Checklist

Before submitting:
- [ ] All cells executed successfully
- [ ] All outputs are visible
- [ ] Training curves show convergence
- [ ] Evaluation metrics are calculated
- [ ] Generated images are displayed
- [ ] File is converted to PDF or HTML
- [ ] File is named correctly: `CV_assignment2_group_4.pdf`

---

## Need Help?

1. **Check README.md** for detailed documentation
2. **Read error messages** carefully
3. **Use discussion forum** for assignment questions
4. **Google the error** - most issues have solutions online

---

## Time Management

| Task | Time | Priority |
|------|------|----------|
| Setup | 10 min | 🔴 Critical |
| Data generation | 5 min | 🔴 Critical |
| Model loading | 5 min | 🔴 Critical |
| Training | 30-60 min | 🔴 Critical |
| Evaluation | 5 min | 🔴 Critical |
| Analysis | 10 min | 🟡 Important |
| Export | 5 min | 🔴 Critical |
| **Total** | **70-100 min** | |

💡 **Pro Tip**: Start early! Run training overnight if needed.

---

## Success Indicators

You're on the right track if you see:
- ✅ Loss decreasing steadily
- ✅ Generated images showing visible text
- ✅ OCR successfully extracting some text
- ✅ Character accuracy > 50%
- ✅ No CUDA errors or crashes

---

**Ready? Let's go! 🚀**

Open `CV_assignment2_group_4.ipynb` and start from Section 2!
