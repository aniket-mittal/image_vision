#!/usr/bin/env python3
"""
Script to compile GroundingDINO CUDA extensions.
Run this after installing GroundingDINO if you encounter '_C not defined' errors.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, cwd=None, check=True):
    """Run a command and return the result."""
    print(f"Running: {cmd}")
    if cwd:
        print(f"  in directory: {cwd}")
    
    result = subprocess.run(
        cmd, 
        shell=True, 
        cwd=cwd, 
        capture_output=True, 
        text=True
    )
    
    if result.stdout:
        print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if check and result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        return False
    
    return True

def compile_extensions():
    """Compile GroundingDINO CUDA extensions."""
    
    # Get the current directory
    current_dir = Path(__file__).parent.absolute()
    groundingdino_dir = current_dir / "Grounded_Segment_Anything" / "GroundingDINO"
    
    if not groundingdino_dir.exists():
        print(f"Error: GroundingDINO directory not found at {groundingdino_dir}")
        return False
    
    print(f"Compiling GroundingDINO extensions in {groundingdino_dir}")
    
    # Check if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA version: {torch.version.cuda}")
        else:
            print("⚠ CUDA not available, will compile CPU-only extensions")
    except ImportError:
        print("⚠ Torch not available, trying to compile anyway...")
    
    # Change to GroundingDINO directory
    os.chdir(groundingdino_dir)
    
    # Try to compile extensions
    print("\n1. Compiling CUDA extensions...")
    if not run_command("python setup.py build_ext --inplace", cwd=groundingdino_dir):
        print("Compilation failed, trying alternative approach...")
        
        # Alternative: try to compile just the C++ extensions
        print("\n2. Trying to compile C++ extensions only...")
        if not run_command("python setup.py build_ext --inplace --force", cwd=groundingdino_dir):
            print("Alternative compilation also failed")
            return False
    
    # Test if the extensions were compiled successfully
    print("\n3. Testing compiled extensions...")
    try:
        # Try to import the compiled extension
        import groundingdino._C
        print("✓ GroundingDINO CUDA extensions compiled successfully!")
        print(f"  Extension location: {groundingdino._C.__file__}")
        return True
    except ImportError as e:
        print(f"⚠ Warning: Could not import compiled extensions: {e}")
        print("  The package will work but may be slower without CUDA acceleration")
        
        # Check if the extension files exist
        csrc_dir = groundingdino_dir / "groundingdino" / "models" / "GroundingDINO" / "csrc"
        if csrc_dir.exists():
            print(f"  Source files found in: {csrc_dir}")
            
            # Look for compiled extensions
            import glob
            so_files = glob.glob(str(groundingdino_dir / "groundingdino" / "*.so"))
            if so_files:
                print(f"  Found compiled extensions: {so_files}")
            else:
                print("  No compiled extensions found")
        
        return False

if __name__ == "__main__":
    success = compile_extensions()
    if success:
        print("\n✅ GroundingDINO extensions compiled successfully!")
    else:
        print("\n❌ GroundingDINO extension compilation failed")
        print("The package may still work but without CUDA acceleration")
    
    sys.exit(0 if success else 1)
