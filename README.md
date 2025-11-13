# MLOps Course - Student Machine Setup

Use this guide to prepare a basic development environment for the first MLOps class.

The first class will focus on working with Jupyter notebooks and converting them into Python scripts, version-controlling your work, and running code reliably in a virtual environment.

## 0) Supported operating systems

- Windows 10/11: use WSL2 (Ubuntu recommended)
- macOS (Apple Silicon or Intel)
- Linux (Ubuntu/Debian-based)

Instructions below include platform-specific steps where needed.

---

## 1) Windows only — Install WSL2 (Ubuntu)

1. Open PowerShell as Administrator and run:
   ```powershell
   wsl --install -d Ubuntu
   ```
   - If prompted, restart your computer.
2. Launch "Ubuntu" from Start Menu, create a UNIX username and password.
3. Update packages inside Ubuntu:
   ```bash
   sudo apt update && sudo apt -y upgrade
   ```

Why WSL2? It provides a Linux environment that matches most production tooling and simplifies Python setup.

---

## 2) Install VS Code

1. Download and install VS Code from the official site: [code.visualstudio.com](https://code.visualstudio.com)
2. Recommended extensions:
   - "Python" (Microsoft)
   - "Jupyter" (Microsoft)
   - "Remote - WSL" (Windows only)

On Windows, after installing "Remote - WSL", open VS Code and connect to Ubuntu: press F1 → "WSL: Connect to WSL".

---

## 3) Install Git

- Windows (inside WSL Ubuntu):
  ```bash
  sudo apt update && sudo apt -y install git
  ```
  (Optional) Install Git for Windows as well: [git-scm.com](https://git-scm.com)

- macOS:
  - If prompted, install Command Line Tools, or install via Homebrew:
    ```bash
    brew install git
    ```

- Linux (Debian/Ubuntu):
  ```bash
  sudo apt update && sudo apt -y install git
  ```

Configure your identity (run in your primary dev shell: WSL Ubuntu/macOS/Linux):
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

(Optional but recommended) Setup SSH for GitHub: follow GitHub docs: [docs.github.com → Generate a new SSH key and add it to GitHub](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)

---

## 4) Create a GitHub account

1. Sign up at: [github.com](https://github.com)
2. Verify your email.
3. (Optional) Enable 2FA for security.

---

## 5) Install Python 3.12 with pyenv

We will use Python 3.12 and manage it with `pyenv`. This keeps your system Python untouched and makes switching versions easy.

### 5.1 Install pyenv

- Windows (WSL Ubuntu):
  ```bash
  # Dependencies
  sudo apt update && sudo apt -y install build-essential curl git zlib1g-dev \
    libssl-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev \
    liblzma-dev tk-dev

  # Install pyenv (via installer)
  curl https://pyenv.run | bash

  # Shell init (Zsh)
  echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
  echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
  echo 'eval "$(pyenv init -)"' >> ~/.zshrc
  exec zsh -l
  ```

- macOS (with Homebrew):
  ```bash
  brew update
  brew install pyenv
  # Shell init (Zsh)
  echo 'eval "$(pyenv init -)"' >> ~/.zshrc
  exec zsh -l
  ```

- Linux (Ubuntu/Debian):
  ```bash
  sudo apt update && sudo apt -y install build-essential curl git zlib1g-dev \
    libssl-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev \
    liblzma-dev tk-dev
  curl https://pyenv.run | bash
  echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
  echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
  echo 'eval "$(pyenv init -)"' >> ~/.bashrc
  exec bash -l
  ```

Verify pyenv is installed:
```bash
pyenv --version
```

### 5.2 Install Python 3.12 and set local version

```bash
pyenv install 3.12.6   # or the latest 3.12.x shown by: pyenv install -l | grep " 3.12"
pyenv global 3.12.6     # or use `pyenv local 3.12.6` inside your course folder
python --version        # should show Python 3.12.x
pip --version
```

---

## 7) Verify Git + GitHub access

1. In your dev shell, generate an SSH key if you chose SSH:
   ```bash
   ssh-keygen -t ed25519 -C "your.email@example.com"
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_ed25519
   ```
2. Add the public key (`~/.ssh/id_ed25519.pub`) to GitHub → Settings → SSH and GPG keys.
3. Test:
   ```bash
   ssh -T git@github.com
   ```
   You should see a success message.

---

## FAQ: Do we need other tools for the first class?

For the first session (Jupyter notebook to Python script), keep it minimal:

- Required:
  - Python 3.12 + `venv`
  - VS Code with Python + Jupyter extensions
  - Jupyter/JupyterLab + ipykernel
  - Git + GitHub account


We will introduce additional tooling (linting, testing, packaging, environment management, containers, CI) in later sessions. For now, please ensure the steps above are completed and that you can run a Jupyter notebook and a simple Python script inside your 3.12 virtual environment.
