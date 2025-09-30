# ComfyUI Apple FastVLM Node

<div align="center">

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![ComfyUI](https://img.shields.io/badge/ComfyUI-compatible-orange.svg)

**Fast and efficient Vision Language Model integration for ComfyUI**

[Installation](#-installation) • [Usage](#-usage) • [Features](#-features) • [Troubleshooting](#-troubleshooting)

</div>

---

## 🌟 Features

- **⚡ Ultra-Fast**: 85x faster Time-to-First-Token compared to LLaVA-OneVision
- **🎯 Multiple Models**: Support for 0.5B, 1.5B, and 7B parameter variants
- **💾 Memory Efficient**: 4-bit and 8-bit quantization support
- **🔄 Smart Caching**: Automatic model caching for faster inference
- **🧹 Clean Output**: Automatic removal of conversation artifacts

## 📋 Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.8+ |
| PyTorch | 2.0+ |
| GPU | CUDA-compatible (recommended) |
| ComfyUI | Latest version |

### System Requirements by Model

| Model | VRAM | RAM | Performance |
|-------|------|-----|-------------|
| FastVLM-0.5B | 4GB+ | 8GB+ | ⚡ Fastest |
| FastVLM-1.5B | 8GB+ | 16GB+ | ⚖️ Balanced |
| FastVLM-7B | 16GB+ | 24GB+ | 🎯 Most Accurate |

> **Note**: 7B model can run on 8GB VRAM with 4-bit quantization

## 🚀 Installation

### Quick Install

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/OBJ-WTF/ComfyUI-AppleFastVLM.git
cd ComfyUI-AppleFastVLM
pip install -r requirements.txt
```

### Install FastVLM

```bash
cd ..
git clone https://github.com/apple/ml-fastvlm.git
cd ml-fastvlm
pip install -e .
cd ../ComfyUI-AppleFastVLM
```

### Download Models

**Automatic download** (recommended):

```bash
# Download lightweight model (recommended to start)
bash get_models.sh 0.5b

# Or download specific model
bash get_models.sh 1.5b  # Balanced model
bash get_models.sh 7b    # Most accurate
bash get_models.sh all   # All models (~12GB total)
```

**Manual download**:

- [FastVLM-0.5B](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage3.zip) - Lightweight, fastest
- [FastVLM-1.5B](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_1.5b_stage3.zip) - Balanced
- [FastVLM-7B](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_7b_stage3.zip) - Most accurate

Extract to `checkpoints/` directory.

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
| **model_path** | Model location | `checkpoints/llava-fastvithd_0.5b_stage3` | Path |
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

## 💡 Example Prompts

### General Description
```
"Describe this image in detail."
"What can you see in this image?"
```

### Specific Questions
```
"How many people are in this image?"
"What colors are dominant in this scene?"
"What is the person in the center doing?"
```

### Object Detection
```
"List all the objects you can identify."
"What animals are visible in this image?"
```

### Text Recognition (OCR)
```
"What text appears in this image?"
"Read the text visible in the scene."
```

### Scene Analysis
```
"What is happening in this image?"
"Describe the setting and atmosphere."
"What time of day does this appear to be?"
```

## ⚙️ Performance Optimization

### For Speed
- Use FastVLM-0.5B model
- Enable 4-bit quantization
- Lower `max_tokens` (128-256)
- Keep temperature at 0.7 or below

### For Accuracy
- Use FastVLM-7B model
- Disable quantization (if VRAM allows)
- Increase `max_tokens` (512-1024)
- Adjust temperature (0.2-0.5 for factual, 0.7-1.0 for creative)

### For Memory Efficiency
- Enable `load_in_4bit` for large models
- Use smaller models when possible
- Close other GPU applications
- Reduce input image resolution if needed

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
1. Model path is correct: `\COMFYUI\custom_nodes\ml-fastvlm\checkpoints/llava-fastvithd_X.Xb_stage3`
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
- Adjust temperature (0.2 for factual, 0.7-1.0 for creative)
- Make prompt more specific
- Ensure image quality is good
- Check if model loaded correctly (no errors in console)

### Slow First Load

**This is normal**. First load takes 30-60 seconds depending on model size and disk speed. Subsequent loads use cached model and are instant.

**To preload model**: Set `force_reload: False` and load once at workflow start.

## 📁 Project Structure

```
ComfyUI-AppleFastVLM/
├── AppleFastVLMNode.py    # Main node implementation
├── __init__.py            # ComfyUI entry point
├── requirements.txt       # Python dependencies
├── get_models.sh          # Model download script
├── README.md              # This file
├── LICENSE                # MIT License
└── checkpoints/           # Models directory (created on first run)
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

**FastVLM Model License**: This node uses Apple's FastVLM. Please review:
- [FastVLM LICENSE](https://github.com/apple/ml-fastvlm/blob/main/LICENSE)
- [FastVLM LICENSE_MODEL](https://github.com/apple/ml-fastvlm/blob/main/LICENSE_MODEL)

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

## 🙏 Credits

- **FastVLM**: [Apple Inc.](https://github.com/apple/ml-fastvlm)
- **LLaVA**: [Haotian Liu et al.](https://github.com/haotian-liu/LLaVA)
- **ComfyUI**: [comfyanonymous](https://github.com/comfyanonymous/ComfyUI)

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/OBJ-WTF/ComfyUI-AppleFastVLM/issues)
- **Discussions**: [GitHub Discussions](https://github.com/OBJ-WTF/ComfyUI-AppleFastVLM/discussions)
- **FastVLM Issues**: [Apple ml-fastvlm](https://github.com/apple/ml-fastvlm/issues)

## 🌟 Star History

If you find this node useful, please consider giving it a star! ⭐

---

<div align="center">
Made with ❤️ for the ComfyUI community
</div>
