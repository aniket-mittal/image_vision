#!/usr/bin/env python3
"""
Simple installation script for GroundingDINO that avoids setup.py issues.
This script installs GroundingDINO in development mode without trying to compile extensions.
"""

import os
import sys
import subprocess
import shutil
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

def install_groundingdino():
    """Install GroundingDINO in development mode."""
    
    # Get the current directory
    current_dir = Path(__file__).parent.absolute()
    groundingdino_dir = current_dir / "Grounded_Segment_Anything" / "GroundingDINO"
    
    if not groundingdino_dir.exists():
        print(f"Error: GroundingDINO directory not found at {groundingdino_dir}")
        return False
    
    print(f"Installing GroundingDINO from {groundingdino_dir}")
    
    # First, try to install with pip in editable mode, but skip deps
    print("\n1. Installing GroundingDINO in editable mode (skipping dependencies)...")
    if not run_command(
        "pip install -e . --no-deps",
        cwd=groundingdino_dir,
        check=False
    ):
        print("Pip install failed, trying alternative approach...")
        
        # Alternative: copy the package directly to site-packages
        print("\n2. Trying alternative installation method...")
        
        # Get site-packages directory
        try:
            import site
            site_packages = site.getsitepackages()[0]
            print(f"Site packages directory: {site_packages}")
        except:
            # Fallback for some environments
            import sys
            site_packages = None
            for path in sys.path:
                if 'site-packages' in path:
                    site_packages = path
                    break
        
        if site_packages:
            # Create a simple __init__.py for groundingdino
            groundingdino_pkg = Path(site_packages) / "groundingdino"
            groundingdino_pkg.mkdir(exist_ok=True)
            
            # Copy the main package
            src_pkg = groundingdino_dir / "groundingdino"
            if src_pkg.exists():
                if groundingdino_pkg.exists():
                    shutil.rmtree(groundingdino_pkg)
                shutil.copytree(src_pkg, groundingdino_pkg)
                print(f"Copied GroundingDINO package to {groundingdino_pkg}")
            else:
                print("Error: groundingdino package directory not found")
                return False
        else:
            print("Error: Could not find site-packages directory")
            return False
    
    # Now install the dependencies separately
    print("\n3. Installing dependencies...")
    requirements_file = groundingdino_dir / "requirements.txt"
    if requirements_file.exists():
        if not run_command(f"pip install -r {requirements_file}"):
            print("Warning: Some dependencies failed to install")
    else:
        print("Warning: requirements.txt not found, installing basic dependencies")
        basic_deps = ["torch", "torchvision", "transformers", "numpy", "opencv-python"]
        for dep in basic_deps:
            if not run_command(f"pip install {dep}", check=False):
                print(f"Warning: Failed to install {dep}")
    
    # Test the installation
    print("\n4. Testing installation...")
    try:
        import groundingdino
        print(f"✓ GroundingDINO imported successfully: {groundingdino.__file__}")
        
        # Try to import some key modules
        try:
            from groundingdino.models import GroundingDINO
            print("✓ GroundingDINO model class imported successfully")
        except ImportError as e:
            print(f"⚠ Warning: Could not import GroundingDINO model: {e}")
        
        try:
            from groundingdino.util import box_ops
            print("✓ GroundingDINO utilities imported successfully")
        except ImportError as e:
            print(f"⚠ Warning: Could not import GroundingDINO utilities: {e}")
            
    except ImportError as e:
        print(f"✗ Error: GroundingDINO import failed: {e}")
        return False
    
    print("\n✅ GroundingDINO installation completed!")
    print("\nNote: If you encounter '_C not defined' errors at runtime,")
    print("you may need to compile the CUDA extensions manually:")
    print(f"  cd {groundingdino_dir}")
    print("  python setup.py build_ext --inplace")
    
    return True

if __name__ == "__main__":
    success = install_groundingdino()
    sys.exit(0 if success else 1)
