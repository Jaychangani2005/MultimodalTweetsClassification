#!/usr/bin/env python3
"""
Launcher script for the Multimodal Tweet Classification Streamlit App
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['streamlit', 'torch', 'transformers', 'PIL']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n📦 Please install requirements:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def check_model_files():
    """Check if model files exist"""
    model_path = Path("E:/notebooks/MultimodalTweetsClassification/models/best_multimodal_informative.pth")
    bert_path = Path("E:/notebooks/MultimodalTweetsClassification/bert_model")
    
    issues = []
    
    if not model_path.exists():
        issues.append(f"❌ Model file not found: {model_path}")
    else:
        print(f"✅ Model file found: {model_path}")
    
    if not bert_path.exists():
        issues.append(f"⚠️  BERT model directory not found: {bert_path}")
        print("   (Will fallback to downloading from Hugging Face)")
    else:
        print(f"✅ BERT model found: {bert_path}")
    
    if issues:
        print("\n🚨 Issues found:")
        for issue in issues:
            print(f"  {issue}")
        print("\nThe app may still work with fallback options.")
        
        choice = input("\nContinue anyway? (y/N): ").lower().strip()
        if choice != 'y':
            return False
    
    return True

def main():
    """Main launcher function"""
    print("🚀 Multimodal Tweet Classification App Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("informative_Attention_graph.py").exists():
        print("❌ Error: Please run this script from the frontend directory")
        print("   Current directory should contain informative_Attention_graph.py")
        sys.exit(1)
    
    print("📁 Working directory: ✅")
    
    # Check requirements
    print("\n🔍 Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    print("📦 Requirements: ✅")
    
    # Check model files
    print("\n🔍 Checking model files...")
    if not check_model_files():
        sys.exit(1)
    
    # Launch the app
    print("\n🎉 All checks passed! Launching Streamlit app...")
    print("\n" + "=" * 50)
    print("📱 The app will open in your default web browser")
    print("🛑 Press Ctrl+C to stop the app")
    print("=" * 50 + "\n")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "informative_Attention_graph.py"])
    except KeyboardInterrupt:
        print("\n\n👋 App stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()