"""
Setup script to install all dependencies and pre-download models.
Run: python setup.py
"""
import subprocess
import sys
from pathlib import Path


def run_command(cmd, desc=""):
    print(f"\n{'='*60}\n{desc or 'Running'}: {cmd}\n{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    if result.stdout:
        print(result.stdout)
    return True


def main():
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("Creating virtual environment...")
        run_command(f"{sys.executable} -m venv .venv", "Creating venv")
    
    # Determine pip path
    if sys.platform == "win32":
        pip_path = ".venv\\Scripts\\pip.exe"
        python_path = ".venv\\Scripts\\python.exe"
    else:
        pip_path = ".venv/bin/pip"
        python_path = ".venv/bin/python"
    
    # Upgrade pip
    run_command(f"{python_path} -m pip install --upgrade pip setuptools wheel", "Upgrading pip")
    
    # Install requirements
    if not Path("requirements.txt").exists():
        print("ERROR: requirements.txt not found!")
        return
    run_command(f"{pip_path} install -r requirements.txt", "Installing requirements")
    
    # Pre-download models (optional, but helpful)
    print("\n" + "="*60)
    print("Pre-downloading models (this may take a while)...")
    print("="*60)
    
    model_scripts = [
        f'{python_path} -c "from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained(\'openai/clip-vit-base-patch32\'); CLIPProcessor.from_pretrained(\'openai/clip-vit-base-patch32\')"',
        f'{python_path} -c "from sentence_transformers import SentenceTransformer; SentenceTransformer(\'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2\')"',
        f'{python_path} -c "from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer; M2M100ForConditionalGeneration.from_pretrained(\'facebook/m2m100_418M\'); M2M100Tokenizer.from_pretrained(\'facebook/m2m100_418M\')"',
    ]
    
    for cmd in model_scripts:
        run_command(cmd, "Downloading models")
    
    print("\n" + "="*60)
    print("Setup complete!")
    print(f"Activate venv: {'".venv\\Scripts\\activate"' if sys.platform == 'win32' else 'source .venv/bin/activate'}")
    print("="*60)


if __name__ == "__main__":
    main()

