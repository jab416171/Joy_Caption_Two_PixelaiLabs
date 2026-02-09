#!/usr/bin/env python3
"""
Joy Caption Remote Server
Runs the complete Joy Caption pipeline as a remote service.

This allows offloading the entire vision + LLM pipeline to a remote server,
maintaining full vision capabilities unlike text-only llama.cpp servers.

Usage:
    python joy_caption_server.py --host 0.0.0.0 --port 8000 --model "Llama-3.1-8B-Lexi-Uncensored-V2-nf4"
"""

import os
import sys
import argparse
import base64
import io
from typing import Optional
from pathlib import Path

# Add parent directory to path to import simple_joy_caption
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, request, jsonify
from PIL import Image
import torch
import numpy as np

# Import Joy Caption components
from simple_joy_caption import (
    SimpleLLMCaptionLoader,
    SimpleLLMCaption,
    build_caption_prompt,
    process_caption_text,
    CAPTION_TYPES,
    CAPTION_LENGTHS
)

app = Flask(__name__)

# Global pipeline storage
pipeline = None
loader = None

def initialize_pipeline(model_name: str):
    """Initialize the Joy Caption pipeline"""
    global pipeline, loader
    
    print(f"🚀 Initializing Joy Caption pipeline with model: {model_name}")
    
    loader = SimpleLLMCaptionLoader()
    pipeline_tuple = loader.load_models(model_name, use_4bit=True)
    pipeline = pipeline_tuple[0]
    
    print(f"✅ Pipeline initialized successfully")
    return pipeline

def pil_to_tensor(pil_image):
    """Convert PIL image to tensor format expected by Joy Caption"""
    # Convert to RGB if needed
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Convert to tensor (B, H, W, C) format
    img_array = np.array(pil_image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_array).unsqueeze(0)  # Add batch dimension
    
    return tensor

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'pipeline_loaded': pipeline is not None
    })

@app.route('/caption', methods=['POST'])
def generate_caption():
    """
    Generate caption for an image.
    
    Expected JSON payload:
    {
        "image": "base64_encoded_image",
        "caption_type": "Descriptive",  # optional
        "caption_length": "medium-length",  # optional
        "lora_trigger": "",  # optional
        "gender_age_replacement": "",  # optional
        "hair_replacement": "",  # optional
        "body_size_replacement": "",  # optional
        "remove_tattoos": false,  # optional
        "remove_jewelry": false  # optional
    }
    
    Returns:
    {
        "caption": "generated caption text",
        "success": true
    }
    """
    global pipeline
    
    if pipeline is None:
        return jsonify({
            'success': False,
            'error': 'Pipeline not initialized'
        }), 500
    
    try:
        data = request.json
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert to tensor
        image_tensor = pil_to_tensor(pil_image)
        
        # Get parameters with defaults
        caption_type = data.get('caption_type', 'Descriptive')
        caption_length = data.get('caption_length', 'medium-length')
        lora_trigger = data.get('lora_trigger', '')
        gender_age_replacement = data.get('gender_age_replacement', '')
        hair_replacement = data.get('hair_replacement', '')
        body_size_replacement = data.get('body_size_replacement', '')
        remove_tattoos = data.get('remove_tattoos', False)
        remove_jewelry = data.get('remove_jewelry', False)
        
        # Generate caption using Joy Caption
        caption_node = SimpleLLMCaption()
        result = caption_node.generate_caption(
            pipeline=pipeline,
            image=image_tensor,
            caption_type=caption_type,
            caption_length=caption_length,
            lora_trigger=lora_trigger,
            gender_age_replacement=gender_age_replacement,
            hair_replacement=hair_replacement,
            body_size_replacement=body_size_replacement,
            remove_tattoos=remove_tattoos,
            remove_jewelry=remove_jewelry
        )
        
        caption = result[0] if isinstance(result, tuple) else result
        
        return jsonify({
            'success': True,
            'caption': caption
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/available_caption_types', methods=['GET'])
def get_caption_types():
    """Get list of available caption types"""
    return jsonify({
        'caption_types': list(CAPTION_TYPES.keys()),
        'caption_lengths': CAPTION_LENGTHS
    })

def main():
    parser = argparse.ArgumentParser(description='Joy Caption Remote Server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--model', default='Llama-3.1-8B-Lexi-Uncensored-V2-nf4', 
                       help='LLM model to use')
    parser.add_argument('--no-auto-download', action='store_true',
                       help='Disable auto-download (model must exist locally)')
    
    args = parser.parse_args()
    
    # Add AUTO-DOWNLOAD prefix if not disabled
    model_name = args.model
    if not args.no_auto_download and not model_name.startswith('AUTO-DOWNLOAD:'):
        model_name = f"AUTO-DOWNLOAD: {model_name}"
    
    # Initialize pipeline
    initialize_pipeline(model_name)
    
    # Start server
    print(f"\n{'='*70}")
    print(f"🌐 Joy Caption Remote Server")
    print(f"   Listening on: http://{args.host}:{args.port}")
    print(f"   Model: {args.model}")
    print(f"{'='*70}\n")
    print(f"Endpoints:")
    print(f"  POST /caption - Generate caption for image")
    print(f"  GET /health - Health check")
    print(f"  GET /available_caption_types - List caption types")
    print(f"{'='*70}\n")
    
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == '__main__':
    main()
