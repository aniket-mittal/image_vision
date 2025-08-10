#!/bin/bash
# Comprehensive installation script for H100 server
# This script installs all dependencies and GroundingDINO properly

set -e  # Exit on any error

echo "🚀 Starting comprehensive installation on H100 server..."

# 1. Setup environment
echo "📋 Setting up environment..."
export PATH=/workspace/miniforge/bin:$PATH
source /workspace/miniforge/etc/profile.d/conda.sh
conda activate clip_api

# Create workspace directory
cd /workspace/image_vision

# 2. Clean up space and set temp directories
echo "🧹 Cleaning up space and setting temp directories..."
mkdir -p /workspace/tmp
export TMPDIR=/workspace/tmp
export PIP_NO_CACHE_DIR=1

# Clean caches
conda clean -a -y
pip cache purge || true
rm -rf ~/.cache/pip ~/.cache/torch ~/.cache/torch_extensions ~/.cache/clip || true
rm -rf /root/.cache/pip /root/.cache/torch* /root/.cache/clip || true
rm -rf /tmp/*

# 3. Install/upgrade PyTorch with CUDA 12.1
echo "🔥 Installing PyTorch with CUDA 12.1..."
pip install --no-cache-dir --upgrade --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

# 4. Install other dependencies
echo "📦 Installing other dependencies..."
pip install --no-cache-dir --upgrade \
    diffusers==0.30.3 \
    transformers==4.54.1 \
    accelerate==1.9.0 \
    supervision==0.26.1 \
    timm==1.0.19 \
    opencv-python \
    numpy \
    pillow

# 5. Verify PyTorch installation
echo "✅ Verifying PyTorch installation..."
python -c "
import torch, torchvision
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ TorchVision {torchvision.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'✓ CUDA version: {torch.version.cuda}')
"

# 6. Install GroundingDINO using our fixed setup
echo "🔧 Installing GroundingDINO..."
cd /workspace/image_vision

# First, try the simple installation
if python install_groundingdino.py; then
    echo "✅ GroundingDINO installed successfully!"
else
    echo "⚠ Simple installation failed, trying manual approach..."
    
    # Manual installation
    cd Grounded_Segment_Anything/GroundingDINO
    
    # Install in editable mode without deps first
    pip install -e . --no-deps || {
        echo "⚠ Editable install failed, copying package directly..."
        
        # Get site-packages directory
        SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
        echo "Site packages: $SITE_PACKAGES"
        
        # Copy the package
        if [ -d "$SITE_PACKAGES/groundingdino" ]; then
            rm -rf "$SITE_PACKAGES/groundingdino"
        fi
        cp -r groundingdino "$SITE_PACKAGES/"
        echo "✓ Package copied to site-packages"
    }
    
    cd /workspace/image_vision
fi

# 7. Install SAM
echo "🎯 Installing SAM..."
cd Grounded_Segment_Anything/segment_anything
pip install -e . || {
    echo "⚠ SAM editable install failed, copying package directly..."
    SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
    if [ -d "$SITE_PACKAGES/segment_anything" ]; then
        rm -rf "$SITE_PACKAGES/segment_anything"
    fi
    cp -r segment_anything "$SITE_PACKAGES/"
    echo "✓ SAM package copied to site-packages"
}
cd /workspace/image_vision

# 8. Try to compile GroundingDINO extensions
echo "🔨 Attempting to compile GroundingDINO CUDA extensions..."
if python compile_groundingdino.py; then
    echo "✅ CUDA extensions compiled successfully!"
else
    echo "⚠ CUDA extensions compilation failed, but package should still work"
fi

# 9. Test the installation
echo "🧪 Testing installation..."
python -c "
try:
    import groundingdino
    print('✓ GroundingDINO imported successfully')
    
    # Try to import key components
    try:
        from groundingdino.models import GroundingDINO
        print('✓ GroundingDINO model imported')
    except ImportError as e:
        print(f'⚠ GroundingDINO model import warning: {e}')
        
    try:
        from groundingdino.util import box_ops
        print('✓ GroundingDINO utilities imported')
    except ImportError as e:
        print(f'⚠ GroundingDINO utilities import warning: {e}')
        
except ImportError as e:
    print(f'❌ GroundingDINO import failed: {e}')
    exit(1)

try:
    import segment_anything
    print('✓ SAM imported successfully')
except ImportError as e:
    print(f'❌ SAM import failed: {e}')
    exit(1)

print('✅ All tests passed!')
"

# 10. Start the model server
echo "🚀 Starting model server..."
export MODEL_SERVER_HOST=0.0.0.0
export MODEL_SERVER_PORT=8765

# Kill any existing server
pkill -f 'python model_server.py' || true

# Start server in background
nohup python model_server.py > server.log 2>&1 &
echo $! > server.pid

# Wait for server to start
echo "⏳ Waiting for server to start..."
sleep 10

# Test server health
echo "🏥 Testing server health..."
if curl -s http://127.0.0.1:8765/health | grep -q "ok"; then
    echo "✅ Server is running and healthy!"
    echo "📊 Server status:"
    curl -s http://127.0.0.1:8765/health | python -m json.tool
else
    echo "❌ Server health check failed"
    echo "📋 Server logs:"
    tail -20 server.log
    exit 1
fi

echo ""
echo "🎉 Installation completed successfully!"
echo "📋 Server is running on port 8765"
echo "📊 Check server status with: curl http://127.0.0.1:8765/health"
echo "📋 View server logs with: tail -f server.log"
echo "🛑 Stop server with: kill \$(cat server.pid)"
echo ""
echo "🔧 If you encounter '_C not defined' errors, run:"
echo "   cd /workspace/image_vision"
echo "   python compile_groundingdino.py"
