# Assignment 2: Fine-Tuning Text-to-Image Model for Text Generation

This project implements **Problem Statement 4** from CV Assignment 2: Fine-tuning Stable Diffusion using LoRA for generating readable text images.

## 📋 Project Overview

The goal is to fine-tune a pre-trained text-to-image model (Stable Diffusion v1.4/v1.5) to generate images containing readable text. This addresses a known limitation of generative models in rendering coherent, legible text within images.

### Key Features
- **Parameter-Efficient Fine-Tuning**: Uses LoRA (Low-Rank Adaptation) to fine-tune only attention layers
- **Comprehensive Evaluation**: OCR-based quantitative metrics (character accuracy, exact match) and qualitative assessment
- **Structured Pipeline**: Complete workflow from data preprocessing to evaluation
- **Well-Documented**: Detailed Jupyter notebook with explanations and visualizations

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 16GB+ RAM
- Tesseract OCR installed on system

### Installation

1. **Clone or navigate to the project directory:**
```bash
cd /Users/sahup3/Projects/MTech/CV_assignment_2
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages:**
```bash
pip install -r requirements.txt
```

4. **Install Tesseract OCR:**
   - **Ubuntu/Debian:** `sudo apt-get install tesseract-ocr`
   - **macOS:** `brew install tesseract`
   - **Windows:** Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

### For Google Colab

If using Google Colab, run these commands at the beginning of your notebook:

```python
# Install packages
!pip install torch torchvision transformers diffusers accelerate peft
!pip install pytesseract easyocr Pillow opencv-python
!pip install numpy pandas matplotlib seaborn tqdm

# Install Tesseract
!apt-get install tesseract-ocr

# Mount Google Drive (optional, for saving models)
from google.colab import drive
drive.mount('/content/drive')
```

## 📁 Project Structure

```
CV_assignment_2/
├── Assignment 2 -CV.pdf          # Original assignment document
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── CV_assignment2_group_4.ipynb  # Main Jupyter notebook
├── data/                         # Generated datasets (created during execution)
│   └── train/                    # Training images and metadata
├── models/                       # Saved model checkpoints (created during training)
│   ├── checkpoint_epoch_10/
│   ├── checkpoint_epoch_20/
│   └── final_lora_model/
└── outputs/                      # Results and visualizations (created during execution)
    ├── training_curves.png
    ├── generated_images_with_ocr.png
    ├── before_after_comparison.png
    ├── performance_metrics.png
    ├── evaluation_results.csv
    └── evaluation_summary.json
```

## 💻 Usage

### Option 1: Run in Jupyter Notebook

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open `CV_assignment2_group_4.ipynb`

3. Run cells sequentially from top to bottom

### Option 2: Run in Google Colab

1. Upload `CV_assignment2_group_4.ipynb` to Google Colab
2. Install dependencies (see instructions above)
3. Run cells sequentially

### Option 3: Export to HTML/PDF for Submission

After running all cells:

**Export to HTML:**
```bash
jupyter nbconvert --to html --execute CV_assignment2_group_4.ipynb
```

**Export to PDF (requires LaTeX):**
```bash
jupyter nbconvert --to pdf --execute CV_assignment2_group_4.ipynb
```

Or use Jupyter's built-in "File → Download as → HTML/PDF" option.

## 📊 Notebook Sections

The notebook is organized into 10 comprehensive sections:

1. **Introduction** - Problem overview and objectives
2. **Environment Setup** - Installation and configuration
3. **Data Collection and Preprocessing** - Dataset generation
4. **Model Architecture and Setup** - Loading Stable Diffusion
5. **LoRA Fine-Tuning Implementation** - Configuring parameter-efficient training
6. **Training Process** - Complete training loop
7. **Evaluation - Quantitative** - OCR-based metrics
8. **Evaluation - Qualitative** - Visual assessment
9. **Results Analysis and Discussion** - Performance analysis
10. **Conclusion** - Summary and future work

## 🎯 Key Technical Details

### Model Configuration
- **Base Model**: Stable Diffusion v1.5 (runwayml/stable-diffusion-v1-5)
- **Resolution**: 512x512 pixels
- **LoRA Rank**: 4
- **LoRA Alpha**: 4
- **Target Modules**: Attention layers (to_q, to_k, to_v, to_out.0)

### Training Configuration
- **Epochs**: 50 (adjustable)
- **Batch Size**: 1 (memory-efficient)
- **Learning Rate**: 1e-4 with cosine scheduler
- **Optimizer**: AdamW
- **Training Samples**: 100 synthetic text images

### Evaluation Metrics
- **Character Accuracy**: Character-level matching between OCR output and ground truth
- **Exact Match Rate**: Percentage of perfectly matched strings
- **Qualitative Criteria**: Visual clarity, font realism, background consistency

## 🔧 Configuration Options

You can modify hyperparameters in the notebook's `Config` class:

```python
class Config:
    # Model settings
    MODEL_ID = "runwayml/stable-diffusion-v1-5"  # or "CompVis/stable-diffusion-v1-4"
    RESOLUTION = 512
    
    # Training settings
    BATCH_SIZE = 1
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    
    # LoRA settings
    LORA_RANK = 4
    LORA_ALPHA = 4
    LORA_DROPOUT = 0.1
```

## 📈 Expected Results

- **Character Accuracy**: 60-85% (depends on text complexity)
- **Exact Match Rate**: 40-70% (for simple, short text)
- **Training Time**: 
  - GPU (T4/V100): ~30-60 minutes
  - CPU: 3-5 hours (not recommended)

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size to 1
   - Use gradient accumulation
   - Use mixed precision training (fp16)

2. **Tesseract Not Found**
   - Install Tesseract binary
   - Set path: `pytesseract.pytesseract.tesseract_cmd = '/path/to/tesseract'`

3. **Model Download Fails**
   - Check internet connection
   - Try using HuggingFace token for private models
   - Use VPN if blocked in your region

4. **Slow Training**
   - Reduce NUM_EPOCHS
   - Reduce training dataset size
   - Use GPU instead of CPU

## 📝 Assignment Submission

### Submission Checklist
- ✅ Jupyter notebook with outputs (HTML or PDF format)
- ✅ All cells executed and outputs visible
- ✅ Properly formatted and aligned outputs
- ✅ Clear comments and documentation
- ✅ File named as: `CV_assignment2_group_4.pdf` or `CV_assignment2_group_4.html`

### Grading Rubric (15 points total)
- **Data Preprocessing** (2.5 pts) - Normalization, resizing, dataset generation
- **Model Development** (5 pts) - LoRA implementation, fine-tuning
- **Evaluation Metrics** (2.5 pts) - OCR-based quantitative evaluation
- **Justification** (2.5 pts) - Results analysis and discussion
- **Documentation & Quality** (2.5 pts) - Code quality, presentation

## 🔬 Experimentation Ideas

Want to improve results? Try:
- Increase training dataset size (200-500 samples)
- Use real crowd-sourced images
- Experiment with different LoRA ranks (8, 16)
- Try different base models (SD v2.1, SD-XL)
- Add more diverse text styles and fonts
- Implement data augmentation
- Use mixed precision training (fp16)
- Fine-tune for more epochs (100-200)

## 📚 References

1. **Stable Diffusion Paper**: "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., CVPR 2022)
2. **LoRA Paper**: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
3. **Hugging Face Diffusers**: https://huggingface.co/docs/diffusers
4. **PEFT Library**: https://huggingface.co/docs/peft

## 📧 Support

For questions about the assignment:
- Use the course discussion forum
- Check assignment instructions in `Assignment 2 -CV.pdf`

## ⚠️ Important Notes

- **Plagiarism**: All work must be original. Zero tolerance policy.
- **Late Submission**: -2 marks penalty
- **Only Latest Submission**: Only the most recent submission will be graded
- **GPU Requirement**: Highly recommended for reasonable training time

## 🎓 Learning Outcomes

By completing this project, you will:
- Understand domain adaptation and model fine-tuning
- Gain experience with parameter-efficient methods (LoRA)
- Learn to work with limited/noisy datasets
- Develop evaluation pipelines for generative models
- Master practical constraints of diffusion model training

---

**Good luck with your assignment! 🚀**
