# ComfyUI Apple FastVLM Node

<div align="center">

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![ComfyUI](https://img.shields.io/badge/ComfyUI-compatible-orange.svg)

**Fast and efficient Vision Language Model integration for ComfyUI**

[Installation](#-installation) • [Usage](#-usage) • [Features](#-features) • [Troubleshooting](#-troubleshooting)

</div>

<img width="1287" height="525" alt="WORKLOAD" src="https://github.com/user-attachments/assets/48c78503-6ae5-4046-a882-ebc01cc7e4b1" />


---

## ✨ Features

- **⚡ Ultra-Fast**: 85x faster Time-to-First-Token compared to LLaVA-OneVision  
- **🎯 Multiple Models**: Support for 0.5B, 1.5B, and 7B parameter variants  
- **💾 Memory Efficient**: 4-bit and 8-bit quantization support  
- **🔄 Smart Caching**: Automatic model caching for faster inference  

---

## 📋 Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.10+ |
| GPU | CUDA-compatible (recommended) |
| ComfyUI | Latest version |

### System Requirements by Model

| Model | VRAM | RAM | Performance |
|-------|------|-----|-------------|
| FastVLM-0.5B | 4GB+ | 8GB+ | ⚡ Fastest |
| FastVLM-1.5B | 8GB+ | 16GB+ | ⚖️ Balanced |
| FastVLM-7B | 16GB+ | 24GB+ | 🎯 Most Accurate |

> **Note**: 7B model can run on 8GB VRAM with 4-bit quantization  

---

## 🚀 Installation

### Manual Installation

Copy the node and place the folder in `ComfyUI/custom_nodes/`:

```
ComfyUI/custom_nodes/ComfyUI-AppleFastVLM/
```

### Install dependencies

```bash
cd ComfyUI/custom_nodes/ComfyUI-AppleFastVLM
pip install -r requirements.txt
```

If `bitsandbytes` fails to install, disable quantization in the node (`load_in_4bit=False`, `load_in_8bit=False`).  

### Install Apple’s FastVLM library

```bash
cd ..
git clone https://github.com/apple/ml-fastvlm.git
cd ml-fastvlm
pip install -e .
```

### Download models manually

Create a `checkpoints/` folder inside the `ml-fastvlm` folder and place Apple’s Stage 3 weights:

```
checkpoints/
├── llava-fastvithd_0.5b_stage3/
├── llava-fastvithd_1.5b_stage3/
└── llava-fastvithd_7b_stage3/
```

### Models Links

**Manual download**:

- [FastVLM-0.5B](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage3.zip) – Lightweight, fastest  
- [FastVLM-1.5B](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_1.5b_stage3.zip) – Balanced  
- [FastVLM-7B](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_7b_stage3.zip) – Most accurate  

Extract to `checkpoints/` directory.  

---

## 📖 Usage

### In ComfyUI

1. Right-click in the workflow canvas  
2. Navigate to: `Add Node` → `AppleVLM` → `FastVLM`  
3. Connect an image input  
4. Configure parameters  
5. Run the workflow  

### Node Parameters

#### Required Inputs

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| **image** | Input image | - | Any size |
| **prompt** | Instruction text | "Describe this image in detail." | Text |
| **model_path** | Model location | `custom_nodes/ml-fastvlm/checkpoints/llava-fastvithd_0.5b_stage3` | Path |
| **temperature** | Creativity control | 0.7 | 0.0 - 2.0 |
| **max_tokens** | Response length | 256 | 1 - 2048 |

#### Optional Inputs

| Parameter | Description | Default |
|-----------|-------------|---------|
| **load_in_8bit** | Enable 8-bit quantization | False |
| **load_in_4bit** | Enable 4-bit quantization | False |
| **force_reload** | Force model reload | False |

### Temperature Guide

- **0.0 - 0.3**: Deterministic, factual (best for descriptions)  
- **0.4 - 0.7**: Balanced creativity  
- **0.8 - 1.0**: More creative and varied  
- **1.0+**: Highly creative (may be less accurate)  

---

## 🐛 Troubleshooting

### Out of Memory Error

**Symptoms**: `CUDA out of memory` or `RuntimeError`  

**Solutions**:
```bash
# Enable 4-bit quantization
load_in_4bit: True

# Or use a smaller model
model_path: checkpoints/llava-fastvithd_0.5b_stage3

# Or reduce max_tokens
max_tokens: 128
```

### Model Won't Load

**Check**:
1. Model path is correct: `checkpoints/llava-fastvithd_X.Xb_stage3`  
2. Model files are extracted (not still in .zip)  
3. Dependencies installed: `pip install -r requirements.txt`  
4. FastVLM installed: `cd ml-fastvlm && pip install -e .`  

**Verify installation**:
```bash
python -c "from llava.model.builder import load_pretrained_model; print('OK')"
```

### Poor Quality Responses

**Try**:
- Use a larger model (7B instead of 0.5B)  
- Adjust temperature (0.2 for factual, 0.7–1.0 for creative)  
- Make prompt more specific  
- Ensure image quality is good  
- Check if model loaded correctly (no errors in console)  

---

## 📁 Project Structure

```
ComfyUI-AppleFastVLM/
├── AppleFastVLMNode.py    # Main node implementation
├── __init__.py            # ComfyUI entry point
├── requirements.txt       # Python dependencies
├── README.md              # This file
ml-fastvlm
└── checkpoints/           # Models directory 
```

---

## 🤝 Contributing

Contributions are welcome! Please:  

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/amazing-feature`)  
3. Commit your changes (`git commit -m 'Add amazing feature'`)  
4. Push to the branch (`git push origin feature/amazing-feature`)  
5. Open a Pull Request  

---

## 📜 License

This project is licensed under the MIT License – see [LICENSE](LICENSE) file.  

**FastVLM Model License**: This node uses Apple's FastVLM. Please review:  
- [FastVLM LICENSE](https://github.com/apple/ml-fastvlm/blob/main/LICENSE)  
- [FastVLM LICENSE_MODEL](https://github.com/apple/ml-fastvlm/blob/main/LICENSE_MODEL)  

---

## 📚 Citation

If you use FastVLM in your research, please cite:

```bibtex
@InProceedings{fastvlm2025,
    author = {Pavan Kumar Anasosalu Vasu, Fartash Faghri, Chun-Liang Li, 
              Cem Koc, Nate True, Albert Antony, Gokul Santhanam, 
              James Gabriel, Peter Grasch, Oncel Tuzel, Hadi Pouransari},
    title = {FastVLM: Efficient Vision Encoding for Vision Language Models},
    booktitle = {CVPR},
    year = {2025},
}
```

---

## 🙏 Credits

- **FastVLM**: [Apple Inc.](https://github.com/apple/ml-fastvlm)  
- **LLaVA**: [Haotian Liu et al.](https://github.com/haotian-liu/LLaVA)  
- **ComfyUI**: [comfyanonymous](https://github.com/comfyanonymous/ComfyUI)  

---

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/OBJ-WTF/ComfyUI-AppleFastVLM/issues)  
- **Discussions**: [GitHub Discussions](https://github.com/OBJ-WTF/ComfyUI-AppleFastVLM/discussions)  
- **FastVLM Issues**: [Apple ml-fastvlm](https://github.com/apple/ml-fastvlm/issues)  

---

## 🌟 Star History

If you find this node useful, please consider giving it a star! ⭐  

---

<div align="center">
Made with ❤️ for the ComfyUI community
</div>
