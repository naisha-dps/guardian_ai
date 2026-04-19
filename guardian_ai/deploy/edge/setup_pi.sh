#!/bin/bash
# Guardian AI - Raspberry Pi Setup Script
# ==========================================
# Run this on a fresh Raspberry Pi OS (64-bit) installation.
# Usage: chmod +x setup_pi.sh && sudo ./setup_pi.sh

set -e

echo "============================================"
echo "  Guardian AI - Raspberry Pi Setup"
echo "============================================"

# ─── System Update ────────────────────────────────────────────────────────────
echo "[1/8] Updating system..."
apt-get update -y && apt-get upgrade -y

# ─── Install System Dependencies ─────────────────────────────────────────────
echo "[2/8] Installing system packages..."
apt-get install -y \
    python3 python3-pip python3-venv \
    libopencv-dev python3-opencv \
    git curl wget unzip \
    libatlas-base-dev \
    libhdf5-dev \
    libc-ares-dev \
    libeigen3-dev \
    libopenblas-dev \
    i2c-tools \
    python3-serial \
    cmake \
    v4l-utils

# Enable camera and I2C
echo "[3/8] Enabling camera and interfaces..."
raspi-config nonint do_camera 0
raspi-config nonint do_i2c 0
raspi-config nonint do_serial_hw 0

# ─── Create App Directory ─────────────────────────────────────────────────────
echo "[4/8] Setting up app directory..."
mkdir -p /opt/guardian_ai/{models,logs,offline_queue}
mkdir -p /var/guardian_ai/offline_queue

# ─── Clone Repository ─────────────────────────────────────────────────────────
echo "[5/8] Cloning Guardian AI..."
if [ ! -d /opt/guardian_ai/src ]; then
    git clone https://github.com/your-org/guardian-ai.git /opt/guardian_ai/src
fi

# ─── Python Virtual Environment ──────────────────────────────────────────────
echo "[6/8] Setting up Python environment..."
python3 -m venv /opt/guardian_ai/venv
source /opt/guardian_ai/venv/bin/activate

# Install Pi-optimized packages
pip install --upgrade pip wheel
pip install \
    ultralytics \
    onnxruntime \
    opencv-python-headless \
    numpy \
    requests \
    pyserial \
    RPi.GPIO \
    picamera2 \
    websocket-client \
    psutil

# Install OpenVINO for Intel acceleration (optional)
# pip install openvino-dev

# ─── Download Model ───────────────────────────────────────────────────────────
echo "[7/8] Downloading Guardian AI model..."
MODEL_URL="https://your-storage.com/guardian_wildlife_int8.onnx"
if [ ! -f /opt/guardian_ai/models/guardian_wildlife_int8.onnx ]; then
    echo "  Place your trained model at: /opt/guardian_ai/models/guardian_wildlife_int8.onnx"
    echo "  Or run: python /opt/guardian_ai/src/ml/export/export_model.py"
fi

# ─── Systemd Service ──────────────────────────────────────────────────────────
echo "[8/8] Installing systemd service..."
cat > /etc/systemd/system/guardian_ai.service << 'EOF'
[Unit]
Description=Guardian AI Wildlife Detection
After=network.target
Wants=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/opt/guardian_ai/src
Environment=PYTHONPATH=/opt/guardian_ai/src
Environment=BACKEND_URL=http://YOUR_SERVER_IP:8000
Environment=DEVICE_ID=pi_001
ExecStart=/opt/guardian_ai/venv/bin/python edge/raspberry_pi/main_controller.py
Restart=always
RestartSec=10
StandardOutput=append:/var/guardian_ai/logs/guardian_ai.log
StandardError=append:/var/guardian_ai/logs/guardian_ai_error.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable guardian_ai
systemctl start guardian_ai

# ─── Power Optimization ───────────────────────────────────────────────────────
echo "[+] Applying power optimizations..."

# Disable HDMI (saves ~25mA)
tvservice -o

# Set CPU governor to ondemand (balance performance/power)
echo ondemand > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Add to /boot/config.txt
echo "" >> /boot/config.txt
echo "# Guardian AI Power Optimization" >> /boot/config.txt
echo "hdmi_blanking=2" >> /boot/config.txt

# ─── Done ────────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "  Next steps:"
echo "  1. Edit /etc/systemd/system/guardian_ai.service"
echo "     Set BACKEND_URL to your server IP"
echo "  2. Place model at:"
echo "     /opt/guardian_ai/models/guardian_wildlife_int8.onnx"
echo "  3. Restart service:"
echo "     sudo systemctl restart guardian_ai"
echo ""
echo "  View logs: journalctl -u guardian_ai -f"
echo ""
