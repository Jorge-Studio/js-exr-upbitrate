#!/bin/bash
# =============================================================================
# JS-EXR-Upbitrate: Video Model Installer
# Installs AI video generation + upscaling models for ComfyUI
# =============================================================================
set -e

COMFY_DIR="${COMFY_DIR:-/workspace/runpod-slim/ComfyUI}"
MODELS_DIR="$COMFY_DIR/models"
CUSTOM_NODES_DIR="$COMFY_DIR/custom_nodes"

echo "============================================"
echo "JS-EXR Video Pipeline - Model Installer"
echo "ComfyUI: $COMFY_DIR"
echo "============================================"

# Check GPU VRAM
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "GPU Memory: ${GPU_MEM:-unknown} MB"
echo ""

# -----------------------------------------------
# WAN 2.1 Image-to-Video (works on 24GB)
# -----------------------------------------------
install_wan() {
    echo "=== Installing WAN 2.1 I2V ==="
    
    mkdir -p "$MODELS_DIR/diffusion_models" "$MODELS_DIR/clip" "$MODELS_DIR/vae"
    
    # WAN 2.1 I2V 14B (GGUF Q8 for 24GB GPUs)
    if [ ! -f "$MODELS_DIR/diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors" ]; then
        echo "Downloading WAN 2.1 I2V 14B..."
        python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Comfy-Org/Wan_2.1_ComfyUI_repackaged', 'split_files/diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors', local_dir='$MODELS_DIR/..')
" 2>/dev/null && echo "  Downloaded WAN 2.1 I2V 14B" || echo "  Trying alternative..."
    fi
    
    # UMT5-XXL text encoder
    if [ ! -f "$MODELS_DIR/clip/umt5_xxl_fp8_e4m3fn.safetensors" ]; then
        echo "Downloading UMT5-XXL text encoder..."
        python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Comfy-Org/Wan_2.1_ComfyUI_repackaged', 'split_files/text_encoders/umt5_xxl_fp8_e4m3fn.safetensors', local_dir='$MODELS_DIR/..')
" 2>/dev/null && echo "  Downloaded UMT5-XXL" || echo "  Failed"
    fi
    
    # WAN VAE
    if [ ! -f "$MODELS_DIR/vae/wan_2.1_vae.safetensors" ]; then
        echo "Downloading WAN 2.1 VAE..."
        python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Comfy-Org/Wan_2.1_ComfyUI_repackaged', 'split_files/vae/wan_2.1_vae.safetensors', local_dir='$MODELS_DIR/..')
" 2>/dev/null && echo "  Downloaded WAN VAE" || echo "  Failed"
    fi
    
    echo "WAN 2.1 installation complete."
}

# -----------------------------------------------
# LTX-Video 2 (Distilled FP8 for 24GB GPUs)
# -----------------------------------------------
install_ltx2() {
    echo "=== Installing LTX-Video 2 ==="
    
    mkdir -p "$MODELS_DIR/checkpoints" "$MODELS_DIR/clip"
    
    # Install ComfyUI-LTXVideo custom nodes
    if [ ! -d "$CUSTOM_NODES_DIR/ComfyUI-LTXVideo" ]; then
        echo "Installing ComfyUI-LTXVideo nodes..."
        cd "$CUSTOM_NODES_DIR"
        git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git 2>/dev/null || true
        if [ -f "$CUSTOM_NODES_DIR/ComfyUI-LTXVideo/requirements.txt" ]; then
            pip install -r "$CUSTOM_NODES_DIR/ComfyUI-LTXVideo/requirements.txt" 2>/dev/null
        fi
    fi
    
    # LTX-2 distilled FP8 model
    if [ ! -f "$MODELS_DIR/checkpoints/ltx-video-2b-v0.9.5.safetensors" ]; then
        echo "Downloading LTX-Video 2B model..."
        python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Lightricks/LTX-Video', 'ltx-video-2b-v0.9.5.safetensors', local_dir='$MODELS_DIR/checkpoints')
" 2>/dev/null && echo "  Downloaded LTX-Video 2B" || echo "  Failed"
    fi
    
    echo "LTX-Video 2 installation complete."
}

# -----------------------------------------------
# HunyuanVideo (GGUF for 24GB GPUs)
# -----------------------------------------------
install_hunyuan() {
    echo "=== Installing HunyuanVideo ==="
    
    mkdir -p "$MODELS_DIR/diffusion_models" "$MODELS_DIR/clip" "$MODELS_DIR/vae"
    
    # HunyuanVideo model (fp8 for 24GB)
    if [ ! -f "$MODELS_DIR/diffusion_models/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors" ]; then
        echo "Downloading HunyuanVideo fp8..."
        python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Comfy-Org/HunyuanVideo_repackaged', 'split_files/diffusion_models/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors', local_dir='$MODELS_DIR/..')
" 2>/dev/null && echo "  Downloaded HunyuanVideo" || echo "  Failed"
    fi
    
    # CLIP text encoder for HunyuanVideo
    if [ ! -f "$MODELS_DIR/clip/clip-vit-large-patch14.safetensors" ]; then
        echo "Downloading CLIP encoder..."
        python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Comfy-Org/HunyuanVideo_repackaged', 'split_files/text_encoders/clip-vit-large-patch14.safetensors', local_dir='$MODELS_DIR/..')
" 2>/dev/null && echo "  Downloaded CLIP" || echo "  Failed"
    fi
    
    # LLM text encoder
    if [ ! -f "$MODELS_DIR/clip/llava_llama3_fp8_scaled.safetensors" ]; then
        echo "Downloading LLM encoder..."
        python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Comfy-Org/HunyuanVideo_repackaged', 'split_files/text_encoders/llava_llama3_fp8_scaled.safetensors', local_dir='$MODELS_DIR/..')
" 2>/dev/null && echo "  Downloaded LLM encoder" || echo "  Failed"
    fi
    
    # VAE
    if [ ! -f "$MODELS_DIR/vae/hunyuan_video_vae_fp32.safetensors" ]; then
        echo "Downloading HunyuanVideo VAE..."
        python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Comfy-Org/HunyuanVideo_repackaged', 'split_files/vae/hunyuan_video_vae_fp32.safetensors', local_dir='$MODELS_DIR/..')
" 2>/dev/null && echo "  Downloaded VAE" || echo "  Failed"
    fi
    
    echo "HunyuanVideo installation complete."
}

# -----------------------------------------------
# SeedVR2 Video Upscaler
# -----------------------------------------------
install_seedvr2() {
    echo "=== Installing SeedVR2 Upscaler ==="
    
    # Install custom node
    if [ ! -d "$CUSTOM_NODES_DIR/ComfyUI-SeedVR2_VideoUpscaler" ]; then
        echo "Installing SeedVR2 custom nodes..."
        cd "$CUSTOM_NODES_DIR"
        git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git 2>/dev/null || true
        if [ -f "$CUSTOM_NODES_DIR/ComfyUI-SeedVR2_VideoUpscaler/requirements.txt" ]; then
            pip install -r "$CUSTOM_NODES_DIR/ComfyUI-SeedVR2_VideoUpscaler/requirements.txt" 2>/dev/null
        fi
    fi
    
    echo "SeedVR2 installation complete."
    echo "(Model will auto-download on first use)"
}

# -----------------------------------------------
# Copy VHS format files
# -----------------------------------------------
install_vhs_formats() {
    echo "=== Installing VHS Custom Formats ==="
    
    VHS_DIR="$CUSTOM_NODES_DIR/ComfyUI-VideoHelperSuite"
    if [ -d "$VHS_DIR/video_formats" ]; then
        SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
        cp "$SCRIPT_DIR/vhs_formats/"*.json "$VHS_DIR/video_formats/" 2>/dev/null
        echo "  Copied FFV1, ProRes, H.265 HDR10 formats to VHS"
    else
        echo "  VHS not found, skipping format install"
    fi
}

# -----------------------------------------------
# Main
# -----------------------------------------------
echo ""
echo "Select what to install:"
echo "  1) WAN 2.1 I2V (recommended)"
echo "  2) LTX-Video 2"
echo "  3) HunyuanVideo"
echo "  4) SeedVR2 Upscaler"
echo "  5) VHS Format Files"
echo "  A) All of the above"
echo ""

if [ "$1" == "all" ] || [ "$1" == "A" ]; then
    install_wan
    install_ltx2
    install_hunyuan
    install_seedvr2
    install_vhs_formats
else
    read -p "Choice [1-5/A]: " choice
    case $choice in
        1) install_wan ;;
        2) install_ltx2 ;;
        3) install_hunyuan ;;
        4) install_seedvr2 ;;
        5) install_vhs_formats ;;
        [Aa]) install_wan; install_ltx2; install_hunyuan; install_seedvr2; install_vhs_formats ;;
        *) echo "Invalid choice"; exit 1 ;;
    esac
fi

echo ""
echo "============================================"
echo "Installation complete!"
echo "Restart ComfyUI to load new nodes."
echo "============================================"
