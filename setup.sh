#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Always run from the project root (directory of this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${GREEN}🤖 A.X.I.S. Local Agent Setup Script${NC}"
echo -e "${YELLOW}Platform: $(uname -s) | Architecture: $(uname -m)${NC}"

# --- OS Detection ---
detect_os() {
    if [ -f /etc/arch-release ] || command -v pacman &>/dev/null; then
        echo "arch"
    elif command -v apt &>/dev/null; then
        echo "debian"
    else
        echo "unknown"
    fi
}

OS_TYPE=$(detect_os)
ARCH=$(uname -m)
echo -e "${YELLOW}Detected OS type: ${OS_TYPE}${NC}"

# 1. Install System Dependencies
echo -e "${YELLOW}[1/7] Installing system dependencies...${NC}"
if [ "$OS_TYPE" == "arch" ]; then
    echo "  Using pacman (Arch Linux)..."
    sudo pacman -S --needed --noconfirm \
        python tk python-pip \
        alsa-utils portaudio \
        cmake base-devel \
        espeak-ng git wget curl
elif [ "$OS_TYPE" == "debian" ]; then
    echo "  Using apt (Debian/Ubuntu)..."
    sudo apt update
    sudo apt install -y \
        python3-tk libasound2-dev libportaudio2 \
        libatlas-base-dev cmake build-essential \
        espeak-ng git wget curl
else
    echo -e "${RED}⚠️  Unknown OS. Install manually: cmake git python3 portaudio alsa-utils espeak-ng${NC}"
fi

# 2. Create Project Folders
echo -e "${YELLOW}[2/7] Creating project folders...${NC}"
mkdir -p piper
mkdir -p sounds/greeting_sounds sounds/thinking_sounds sounds/ack_sounds sounds/error_sounds
mkdir -p faces/idle faces/listening faces/thinking faces/speaking faces/error faces/capturing faces/warmup

# 3. Download Piper TTS Binary (supports aarch64 and x86_64)
echo -e "${YELLOW}[3/7] Setting up Piper TTS binary...${NC}"
if [ ! -f "piper/piper" ]; then
    PIPER_URL=""
    if [ "$ARCH" == "aarch64" ]; then
        PIPER_URL="https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz"
    elif [ "$ARCH" == "x86_64" ]; then
        PIPER_URL="https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz"
    else
        echo -e "${RED}⚠️  Unsupported architecture: $ARCH. Skipping Piper binary download.${NC}"
    fi

    if [ -n "$PIPER_URL" ]; then
        echo "  Downloading Piper for $ARCH..."
        wget -O /tmp/piper.tar.gz "$PIPER_URL"
        tar -xf /tmp/piper.tar.gz -C piper --strip-components=1
        rm /tmp/piper.tar.gz
        chmod +x piper/piper 2>/dev/null || true
        echo -e "${GREEN}  Piper installed at piper/piper${NC}"
    fi
else
    echo -e "${GREEN}  Piper binary already present.${NC}"
fi

# 4. Download Piper Voice Model (Spanish: es_ES-davefx-medium)
echo -e "${YELLOW}[4/7] Downloading voice model (español)...${NC}"
PIPER_VOICE="es_ES-davefx-medium"
if [ ! -f "piper/${PIPER_VOICE}.onnx" ]; then
    wget -O "piper/${PIPER_VOICE}.onnx" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx"
    wget -O "piper/${PIPER_VOICE}.onnx.json" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json"
    echo -e "${GREEN}  Modelo de voz descargado: ${PIPER_VOICE}${NC}"
else
    echo -e "${GREEN}  Modelo de voz ya presente: ${PIPER_VOICE}${NC}"
fi

# 5. Build whisper.cpp (STT engine)
echo -e "${YELLOW}[5/7] Building whisper.cpp (speech-to-text)...${NC}"
if [ ! -f "whisper.cpp/build/bin/whisper-cli" ]; then
    if [ ! -d "whisper.cpp" ]; then
        echo "  Cloning whisper.cpp..."
        git clone https://github.com/ggerganov/whisper.cpp whisper.cpp
    fi

    echo "  Building whisper.cpp (this may take a few minutes)..."
    cmake -B whisper.cpp/build -S whisper.cpp
    cmake --build whisper.cpp/build --config Release -j"$(nproc)"

    if [ -f "whisper.cpp/build/bin/whisper-cli" ]; then
        echo -e "${GREEN}  whisper-cli built successfully.${NC}"
    else
        echo -e "${RED}❌ whisper-cli build FAILED. Check cmake output above.${NC}"
    fi
else
    echo -e "${GREEN}  whisper-cli already built.${NC}"
fi

# Download Whisper models
# ggml-small.bin (~244 MB) — recommended for Spanish, much better accuracy than base
# ggml-base.bin  (~74 MB)  — fallback, lower accuracy
mkdir -p whisper.cpp/models

echo -e "${YELLOW}  Descargando modelo Whisper (ggml-small — recomendado para español)...${NC}"
if [ ! -f "whisper.cpp/models/ggml-small.bin" ]; then
    wget -O whisper.cpp/models/ggml-small.bin \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin"
    echo -e "${GREEN}  Modelo ggml-small descargado.${NC}"
else
    echo -e "${GREEN}  ggml-small ya presente.${NC}"
fi

# Keep base model as fallback (small download)
echo -e "${YELLOW}  Descargando modelo Whisper (ggml-base — fallback)...${NC}"
if [ ! -f "whisper.cpp/models/ggml-base.bin" ]; then
    wget -O whisper.cpp/models/ggml-base.bin \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin"
    echo -e "${GREEN}  Modelo ggml-base descargado.${NC}"
else
    echo -e "${GREEN}  ggml-base ya presente.${NC}"
fi

# 6. Python Virtual Environment + Dependencies
echo -e "${YELLOW}[6/7] Installing Python libraries...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 7. Ollama AI Models + Wake Word
echo -e "${YELLOW}[7/7] Checking AI models (Ollama)...${NC}"
if command -v ollama &>/dev/null; then
    ollama pull gemma3:1b
    ollama pull moondream
else
    echo -e "${RED}❌ Ollama not found. Install from: https://ollama.com${NC}"
fi

if [ ! -f "wakeword.onnx" ]; then
    echo "  Downloading default 'Hey Jarvis' wake word model..."
    curl -L -o wakeword.onnx \
        "https://github.com/dscripka/openWakeWord/raw/main/openwakeword/resources/models/hey_jarvis_v0.1.onnx"
fi

# --- Final Verification ---
echo ""
echo -e "${YELLOW}--- Verification ---${NC}"
ERRORS=0

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}  ✓ $2${NC}"
    else
        echo -e "${RED}  ✗ MISSING: $2 ($1)${NC}"
        ERRORS=$((ERRORS + 1))
    fi
}

check_file "whisper.cpp/build/bin/whisper-cli"    "whisper-cli binary"
check_file "whisper.cpp/models/ggml-small.bin"    "whisper model (ggml-small — recomendado para español)"
check_file "whisper.cpp/models/ggml-base.bin"     "whisper model (ggml-base — fallback)"
check_file "piper/piper"                           "piper TTS binary"
check_file "piper/es_ES-davefx-medium.onnx"        "piper voice model (español)"
check_file "wakeword.onnx"                         "wake word model"

echo ""
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✨ Setup completo. Ejecuta:${NC}"
    echo -e "${GREEN}   source venv/bin/activate && python axis.py${NC}"
else
    echo -e "${RED}⚠️  Setup completado con ${ERRORS} componente(s) faltante(s). Revisa arriba.${NC}"
    echo -e "${YELLOW}   Cuando estén listos: source venv/bin/activate && python axis.py${NC}"
fi
