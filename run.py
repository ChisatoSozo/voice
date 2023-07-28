import subprocess
import sys


def check_pip_installed():
    try:
        # Check if pip is installed by trying to import it
        import pip
        print("pip is already installed.")
    except ImportError:
        print("pip is not installed. Installing pip...")
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
        print("pip has been installed successfully.")

def install_requirements():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements have been installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while installing requirements: {e}")
        sys.exit(1)

def run_clock_script():
    try:
        subprocess.check_call([sys.executable, "clock.py"])
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running clock.py: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_pip_installed()
    install_requirements()
    run_clock_script()