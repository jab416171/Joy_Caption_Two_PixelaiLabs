# Joy Caption Two - PixelaiLabs Edition

<div align="center">

**Automated Image Captioning for ComfyUI with Joy Caption Alpha Two**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-orange)](https://github.com/comfyanonymous/ComfyUI)

**Created by [PixelaiLabs.com](https://pixelailabs.com) | [@aiconomist](https://youtube.com/@aiconomist)**

[Installation](#installation) • [Features](#features) • [Usage](#usage) • [License](#license)

</div>


## 🔔 Latest Updates

### February 2026 - Remote Joy Caption Server
- ✅ **Full Remote Pipeline** - Run complete Joy Caption (vision + LLM) on remote servers
- ✅ **Identical Quality** - Remote mode maintains full vision capabilities
- ✅ **VRAM Optimization** - Offload everything to powerful remote machines
- ✅ **Easy Setup** - Simple Flask-based server included
- 🎯 **Perfect for distributed setups** - Share GPU servers across multiple workstations

### November 6, 2025 - VRAM Management Update
- ✅ **Automatic VRAM Management** - Models now automatically unload from VRAM after each caption generation
- ✅ **Smart Reloading** - Models reload to GPU only when needed for the next caption
- ✅ **Frees ~10GB VRAM** - VRAM is released between caption generations, preventing memory buildup
- ✅ **No Manual Intervention** - Everything happens automatically in the background
- 🎯 **Perfect for batch processing** - Process large datasets without running out of memory

---

## 🌟 What is This?

A **fully automated** ComfyUI custom node that uses **Joy Caption Alpha Two** to generate high-quality image captions. Perfect for:
- LoRA/model training dataset preparation
- Image tagging and organization
- NSFW content captioning
- Batch processing large image collections

### Why This Version?

- ✅ **100% Automated** - All models download automatically on first use
- ✅ **Joy Caption Quality** - Uses the exact same LoRA adapter and process as the original
- ✅ **Clean Output** - No "Here is a caption:" prefixes or numbered lists
- ✅ **Training Ready** - Batch node outputs chronologically numbered files (1.png, 1.txt, 2.png, 2.txt, ...)
- ✅ **Easy Setup** - Just install requirements and run!

---

## 📦 Installation

### Step 1: Clone the Repository

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Pixelailabs/Joy_Caption_Two_PixelaiLabs.git
```

### Step 2: Install Requirements

```bash
cd Joy_Caption_Two_PixelaiLabs
pip install -r requirements.txt
```

### Step 3: Restart ComfyUI

That's it! On first use, the node will automatically download:
- **SigLIP vision model** (~1.5GB)
- **Joy Caption adapters** (~2.5GB total):
  - Image adapter (86 MB)
  - Custom CLIP weights (1.71 GB)
  - LoRA adapter (671 MB) - *This ensures clean caption output!*
- **Your chosen LLM** (~4-5GB when you select it)

**Total first-time download: ~6-8GB** (one-time only!)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Joy Caption LoRA** | Uses the official 671MB LoRA adapter for professional captions |
| **NSFW Capable** | Handles adult content with image embeddings (LLM truly "sees" the image) |
| **Automated Setup** | All models download automatically - no manual file placement |
| **Multiple Caption Styles** | Descriptive, Training Prompt, Booru Tags, Art Critic, etc. |
| **Adjustable Length** | From very short to very long captions |
| **Text Processing** | Replace gender/age, hair, body type, remove tattoos/jewelry |
| **Batch Processing** | Process entire folders with chronological naming for training |
| **Dual Outputs** | Advanced node outputs positive + negative prompts |
| **Remote Server Support** | Run complete pipeline on remote servers with full vision capability |

---

## 🌐 Remote Joy Caption Server

**NEW: Offload the entire Joy Caption pipeline to a remote server!**

This feature allows you to run the complete vision + LLM pipeline on a separate server (e.g., a more powerful machine or cloud instance), maintaining **FULL vision capabilities and caption quality**.

### ✅ Full Vision Support

Unlike basic llama.cpp servers, our remote solution runs the **COMPLETE Joy Caption pipeline**:
- ✅ **SigLIP Vision Processing** - Image understanding happens on remote server
- ✅ **Image Adapter** - Vision-to-language translation on remote server
- ✅ **LLM with LoRA** - Caption generation with full visual context
- ✅ **Identical Quality** - Remote mode produces the SAME high-quality output as local mode

### Setting Up Remote Server

1. **On your remote server** (GPU machine, cloud instance, etc.):
   ```bash
   cd Joy_Caption_Two_PixelaiLabs
   pip install -r requirements.txt
   python joy_caption_server.py --host 0.0.0.0 --port 8000 --model "Llama-3.1-8B-Lexi-Uncensored-V2-nf4"
   ```

2. **On your ComfyUI client** machine:
   - Add "Simple LLM Caption Loader" node
   - Enable `use_remote_server` checkbox
   - Set `remote_server_url` to your server (e.g., `http://192.168.1.100:8000`)
   - Connect and generate captions!

### Benefits of Remote Mode

- 🚀 **Offload GPU Work** - Free up local VRAM completely (vision + LLM run remotely)
- ⚡ **Powerful Hardware** - Use cloud GPUs or dedicated servers
- 🔄 **Multi-User** - Multiple ComfyUI instances can share one server
- 💰 **Cost Effective** - Rent GPU instances only when needed
- 📡 **Network Efficiency** - Only images and captions are transmitted (not embeddings)

### Server API

The Joy Caption Server provides a simple REST API:

**POST /caption**
```json
{
  "image": "base64_encoded_image_data",
  "caption_type": "Descriptive",
  "caption_length": "medium-length",
  "lora_trigger": "",
  ...
}
```

**Response**
```json
{
  "success": true,
  "caption": "A young woman with long blonde hair..."
}
```

---
---

## 🚀 Usage

### Basic Workflow

```
[Load Image] → [Simple LLM Caption Loader] → [Simple LLM Caption] → [Save Text]
                                                      ↓
                                                  Caption Text
```

1. **Add "Simple LLM Caption Loader" node**
   - Select LLM: "AUTO-DOWNLOAD: Llama-3.1-8B-Lexi-Uncensored-V2-nf4" (recommended)
   - Models download automatically on first run

2. **Add "Simple LLM Caption" node**
   - Connect pipeline from loader
   - Connect your image
   - Choose caption type and length
   - Optional: Add LoRA trigger, text replacements

3. **Output:** Clean caption text ready to use!

### Batch Processing for Training

Perfect for preparing LoRA/model training datasets:

```
[Simple LLM Caption Loader] → [Simple LLM Caption Batch]
                                       ↓
                              Chronologically numbered files
```

**Input folder:**
```
my_photos/
  ├── photo1.jpg
  ├── image_abc.png
  └── IMG_5678.jpg
```

**Output folder:**
```
training_data/
  ├── 1.png  ← photo1.jpg converted
  ├── 1.txt  ← caption for 1.png
  ├── 2.png  ← image_abc.png converted
  ├── 2.txt  ← caption for 2.png
  ├── 3.png  ← IMG_5678.jpg converted
  └── 3.txt  ← caption for 3.png
```

Ready to use with Kohya, aitoolkit, or any training tool!

---

## 🎨 Caption Types

| Type | Description |
|------|-------------|
| **Descriptive** | Formal, detailed description |
| **Descriptive (Informal)** | Casual, conversational description |
| **Training Prompt** | Stable Diffusion style prompt |
| **MidJourney** | MidJourney style prompt |
| **Booru Tags** | Tag list (Danbooru/Gelbooru format) |
| **Art Critic** | Detailed analysis of composition and style |
| **Social Media** | Engaging caption for social posts |

---

## ⚙️ Nodes

### 1. Simple LLM Caption Loader
Loads the LLM and Joy Caption adapter models (local or remote).

**Parameters:**
- `llm_model` - Choose from dropdown (AUTO-DOWNLOAD options shown first)
  - Recommended: "AUTO-DOWNLOAD: Llama-3.1-8B-Lexi-Uncensored-V2-nf4"
  - Only shows Llama-based models (Joy Caption compatible)
  - Automatically filters out incompatible models (Florence, CLIP, etc.)
- `use_4bit` - Enable 4-bit quantization (recommended for 8GB VRAM, local mode only)
- `use_remote_server` - Enable remote Joy Caption server mode (optional)
- `remote_server_url` - URL of remote Joy Caption server (default: http://localhost:8000)

**Note:** Vision model (SigLIP) downloads automatically in local mode - no selection needed!

### 2. Simple LLM Caption
Generate captions for single images.

**Parameters:**
- `caption_type` - Style of caption
- `caption_length` - Target length (any, short, medium, long, very long)
- `lora_trigger` - Text to prepend to caption
- Text replacements (gender/age, hair, body size)
- Filters (remove tattoos, remove jewelry)

**Output:** `STRING` (caption text)

### 3. Simple LLM Caption Advanced
Advanced options with positive/negative prompt outputs.

**Additional Parameters:**
- `temperature` / `top_p` - Control randomness
- `max_new_tokens` - Maximum caption length
- `append_to_caption` - Text to add at END of caption
- `negative_prompt` - Manual negative prompt input

**Outputs:** 
- `positive_prompt` (STRING) - Final caption
- `negative_prompt` (STRING) - Your negative prompt

### 4. Simple LLM Caption Batch
Batch process folders with chronological naming for training.

**Parameters:**
- `input_directory` - Folder with images
- `output_directory` - Where to save (REQUIRED)
- All caption options from basic node

**Output:** Status message

---

## 🔧 System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **VRAM** | 8GB (with 4-bit) | 12GB+ |
| **RAM** | 16GB | 32GB+ |
| **Storage** | 10GB free | 20GB+ free |
| **Python** | 3.10+ | 3.11 |

---

## 📖 Credits & Attribution

This custom node is built using:

### Core Technology
- **[Joy Caption](https://github.com/fpgaminer/joycaption)** by fpgaminer - Original Joy Caption implementation
- **[Joy Caption Alpha Two](https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two)** by fancyfeast - LoRA adapter and training

### Inspiration
- **[ComfyUI_SLK_joy_caption_two](https://github.com/EvilBT/ComfyUI_SLK_joy_caption_two)** by EvilBT - ComfyUI implementation reference

### Models Used
- **SigLIP** by Google - Vision backbone
- **Llama 3.1** by Meta - Language model

---

## 📄 License

This project is licensed under the **GNU General Public License v3.0** - see the [LICENSE](LICENSE) file for details.

**Important Notes:**
- This is a derivative work based on Joy Caption Alpha Two (GPL-3.0)
- Commercial use is allowed under GPL-3.0 terms
- Any modifications must also be released under GPL-3.0
- Please provide attribution when using or modifying this code

---

## 🤝 Support

- **Issues:** [GitHub Issues](https://github.com/YOUR_USERNAME/Joy_Caption_Two_PixelaiLabs/issues)
- **YouTube:** [@aiconomist](https://youtube.com/@aiconomist)
- **Website:** [PixelaiLabs.com](https://pixelailabs.com)

---

## 🎯 Roadmap

- [ ] Support for more LLM models
- [ ] Custom LoRA adapter training
- [ ] Multi-language captions
- [ ] Video frame captioning

---

<div align="center">

**Made with ❤️ by [PixelaiLabs.com](https://pixelailabs.com)**

If you find this useful, please ⭐ star the repository!

</div>
