# Image Vision

A comprehensive AI-powered image analysis and editing platform that combines multiple computer vision models for intelligent image understanding and manipulation.

## What It Does

**Image Analysis & Chat**: Upload images and chat with AI agents about what's in them. The system can detect objects, describe scenes, and answer questions about image content.

**Smart Object Detection**: Automatically identifies and segments objects in images using GroundingDINO + SAM (Segment Anything Model) for precise object boundaries.

**Attention Mask Generation**: Uses CLIP models to generate attention masks based on text queries, highlighting specific regions of interest in images.

**Interactive Editing Tools**: 
- Crop and lasso selection tools
- Object-aware editing with AI assistance
- Lighting and enhancement controls
- Inpainting and background manipulation

**Multi-Model AI Support**: Integrates with various AI models including GPT, Claude, Gemini, LLaVA, and more for comprehensive image understanding.

## Key Features

- **Web Interface**: Modern React-based frontend with real-time image processing
- **Model Server**: Persistent Python backend that loads models once for faster inference
- **Object Segmentation**: Precise object detection and masking
- **Attention Visualization**: CLIP-based attention maps for understanding AI focus
- **Batch Processing**: Support for multiple image processing workflows
- **Export Options**: Save processed images with various enhancement settings

## Use Cases

- **Content Creation**: Remove backgrounds, enhance lighting, edit specific objects
- **Research & Analysis**: Understand what AI models focus on in images
- **Image Understanding**: Get detailed descriptions and analysis of visual content
- **Educational**: Learn about computer vision and AI image processing

## Architecture

- **Frontend**: Next.js with TypeScript and shadcn/ui components
- **Backend**: Python model server with CLIP, SAM, and GroundingDINO
- **Models**: Pre-trained vision models for object detection and segmentation
- **API**: RESTful endpoints for image processing and AI interactions

This project demonstrates advanced computer vision capabilities by combining multiple state-of-the-art models into a user-friendly web application for intelligent image analysis and editing.
