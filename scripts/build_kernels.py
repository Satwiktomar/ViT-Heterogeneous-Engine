import os
import subprocess
import sys

def build():
    print("⚙️ Triggering C++ Build Process...")
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    setup_path = os.path.join(root_dir, "setup.py")
    
    # Run python setup.py install
    subprocess.check_call([sys.executable, setup_path, "install"])
    print("✅ Build command finished.")

if __name__ == "__main__":
    build()