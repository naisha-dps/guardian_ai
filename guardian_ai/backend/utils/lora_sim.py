"""
Guardian AI - LoRa / GSM Alert Simulator
==========================================
Simulates LoRa and GSM SMS alerts for remote areas
where internet connectivity is unavailable.

Real implementation would use:
  - LoRa: RFM95W module + python-sx127x library
  - GSM: SIM800L module + AT commands via serial
"""

import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

# Animal alert messages in multiple languages
ALERT_MESSAGES = {
    "deer":   {"en": "Deer detected!", "hi": "हिरण पाया गया!", "mr": "हरिण आढळला!"},
    "boar":   {"en": "Wild boar detected!", "hi": "जंगली सूअर!", "mr": "रानडुक्कर!"},
    "wolf":   {"en": "Wolf detected! URGENT", "hi": "भेड़िया! तत्काल", "mr": "लांडगा! तातडीचे"},
    "cattle": {"en": "Stray cattle detected", "hi": "आवारा मवेशी", "mr": "भटकी जनावरे"},
    "dog":    {"en": "Dog detected", "hi": "कुत्ता पाया गया", "mr": "कुत्रा आढळला"},
}


class LoRaSimulator:
    """
    Simulates LoRa radio transmission for remote alerts.
    
    In production, replace _lora_transmit() with:
      import sx127x
      lora = sx127x.LoRa(...)
      lora.send(bytes(message, 'utf-8'))
    """

    def __init__(self, language: str = "hi"):
        self.language = language
        self.message_count = 0

    async def send_alert(self, class_name: str, device_id: str):
        """Send LoRa alert for detected animal."""
        messages = ALERT_MESSAGES.get(class_name, {})
        msg = messages.get(self.language, messages.get("en", "Animal detected"))
        full_msg = f"[{device_id}] {msg} @ {datetime.utcnow().strftime('%H:%M')}"

        # Simulate transmission delay (LoRa is slow ~250-5000 bps)
        await asyncio.sleep(0.1)
        await self._lora_transmit(full_msg)

        # Also simulate SMS fallback via GSM
        await self._gsm_sms(full_msg)

        self.message_count += 1
        logger.info(f"[LoRa/GSM] Sent #{self.message_count}: {full_msg}")

    async def _lora_transmit(self, message: str):
        """
        SIMULATION: Replace with real LoRa transmission.
        
        Real code:
          import serial
          ser = serial.Serial('/dev/ttyUSB0', 9600)
          # LoRa AT commands (e.g., RYLR998 module)
          ser.write(f'AT+SEND=0,{len(message)},{message}\r\n'.encode())
        """
        logger.debug(f"[LoRa SIM] TX: {message}")

    async def _gsm_sms(self, message: str, to: str = "+91XXXXXXXXXX"):
        """
        SIMULATION: Replace with real GSM SMS via SIM800L.
        
        Real code:
          import serial
          ser = serial.Serial('/dev/ttyAMA0', 9600)
          ser.write(b'AT+CMGF=1\r\n')   # SMS text mode
          ser.write(f'AT+CMGS="{to}"\r\n'.encode())
          ser.write(f'{message}\x1A'.encode())  # Ctrl+Z to send
        """
        logger.debug(f"[GSM SIM] SMS to {to}: {message}")
