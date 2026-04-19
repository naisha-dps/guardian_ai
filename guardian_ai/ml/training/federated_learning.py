"""
Guardian AI - Federated Learning (Future Scope)
================================================
Proof-of-concept for privacy-preserving federated training.

Architecture:
  - Each Raspberry Pi trains locally on its own camera data
  - Only model gradients (NOT images) are sent to server
  - Server aggregates updates using FedAvg algorithm
  - Updated global model pushed back to all devices

This enables:
  - Privacy: Farmer's field images never leave the device
  - Personalization: Each device adapts to local wildlife patterns
  - Continuous learning: Model improves from real field data
  - Offline training: Works without constant internet

Usage (future):
  # On server:
  python federated_server.py --rounds 10 --min_clients 3
  
  # On each Pi:
  python federated_client.py --server_url http://SERVER_IP:9000
"""

import copy
import json
import logging
import threading
from typing import List, Dict, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ─── FedAvg Algorithm ────────────────────────────────────────────────────────

def federated_average(
    global_weights: Dict[str, torch.Tensor],
    client_updates: List[Tuple[Dict[str, torch.Tensor], int]],
) -> Dict[str, torch.Tensor]:
    """
    Federated Averaging (FedAvg) - McMahan et al., 2017.
    
    Aggregates model updates from multiple clients using
    weighted averaging by dataset size.
    
    Args:
        global_weights: Current global model weights
        client_updates: List of (weight_delta, num_samples) per client
    
    Returns:
        Updated global model weights
    """
    total_samples = sum(n for _, n in client_updates)
    
    # Initialize aggregated update with zeros
    aggregated = {k: torch.zeros_like(v) for k, v in global_weights.items()}
    
    for client_weights, num_samples in client_updates:
        # Weight each client's contribution proportionally to its dataset size
        weight = num_samples / total_samples
        for key in aggregated:
            if key in client_weights:
                aggregated[key] += weight * client_weights[key]
    
    return aggregated


# ─── Federated Server ────────────────────────────────────────────────────────

class FederatedServer:
    """
    Central server for federated learning coordination.
    
    Responsibilities:
    1. Distribute global model to all Pi clients
    2. Collect gradient updates from clients
    3. Aggregate updates using FedAvg
    4. Update and redistribute improved model
    """

    def __init__(self, model_path: str, port: int = 9000):
        self.model_path = model_path
        self.port = port
        self.round_number = 0
        self.client_updates = []
        self.min_clients = 3        # Wait for at least 3 Pi devices
        self.lock = threading.Lock()
        
        # Load base model
        logger.info(f"Loading base model: {model_path}")

    def receive_update(self, device_id: str, gradient_update: dict, num_samples: int):
        """
        Receive gradient update from a Pi client.
        Gradients arrive as compressed diffs (NOT raw images).
        """
        with self.lock:
            self.client_updates.append({
                "device_id": device_id,
                "update": gradient_update,
                "num_samples": num_samples,
                "round": self.round_number,
            })
            logger.info(
                f"Received update from {device_id} "
                f"({num_samples} samples, round {self.round_number})"
            )

            # Trigger aggregation when enough clients have submitted
            if len(self.client_updates) >= self.min_clients:
                self._aggregate_and_update()

    def _aggregate_and_update(self):
        """Run FedAvg and update global model."""
        logger.info(f"Aggregating {len(self.client_updates)} client updates...")

        # Extract weights and sample counts
        updates = [
            (c["update"], c["num_samples"]) 
            for c in self.client_updates
        ]

        # NOTE: In real implementation, load actual model weights here
        # global_weights = self._load_model_weights()
        # new_weights = federated_average(global_weights, updates)
        # self._save_model_weights(new_weights)

        self.round_number += 1
        self.client_updates.clear()

        logger.info(
            f"Aggregation complete. Global model updated to round {self.round_number}"
        )

    def get_global_model_info(self) -> dict:
        """Return current global model metadata for clients to download."""
        return {
            "round": self.round_number,
            "model_url": f"http://server:{self.port}/model/latest",
            "model_hash": "sha256:placeholder",
        }


# ─── Federated Client (runs on Raspberry Pi) ─────────────────────────────────

class FederatedClient:
    """
    Client running on Raspberry Pi.
    
    Responsibilities:
    1. Download latest global model
    2. Fine-tune locally on new field images
    3. Compute gradient update (model delta)
    4. Send ONLY gradients to server (NOT images)
    """

    def __init__(
        self,
        device_id: str,
        server_url: str,
        local_data_dir: str = "/var/guardian_ai/local_data",
    ):
        self.device_id = device_id
        self.server_url = server_url
        self.local_data_dir = Path(local_data_dir)
        self.local_data_dir.mkdir(parents=True, exist_ok=True)

    def collect_local_sample(self, image, label: dict):
        """
        Save a new labeled sample for local training.
        Called when a detection is confirmed by the farmer.
        
        The farmer can confirm: "Yes, this was a wolf" → saved for training
        """
        import cv2
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = self.local_data_dir / f"img_{timestamp}.jpg"
        lbl_path = self.local_data_dir / f"img_{timestamp}.json"

        cv2.imwrite(str(img_path), image)
        with open(lbl_path, "w") as f:
            json.dump(label, f)

        logger.info(f"Local sample saved: {img_path}")

    def compute_local_update(
        self,
        global_model_path: str,
        local_epochs: int = 5,
        learning_rate: float = 0.001,
    ) -> Tuple[dict, int]:
        """
        Fine-tune global model on local data.
        Returns gradient delta (not full weights) and sample count.
        
        Privacy guarantee: Only ∆weights sent to server,
        never the actual images.
        """
        local_samples = list(self.local_data_dir.glob("*.jpg"))
        num_samples = len(local_samples)

        if num_samples < 10:
            logger.warning(f"Only {num_samples} local samples. Need ≥10 for update.")
            return {}, 0

        logger.info(
            f"Computing local update on {num_samples} samples, "
            f"{local_epochs} epochs..."
        )

        # NOTE: In real implementation:
        # 1. Load global model
        # 2. Fine-tune on local data
        # 3. Compute delta = new_weights - global_weights
        # 4. Optionally: apply differential privacy noise to delta
        # 5. Compress delta (quantize/prune small values)
        # 6. Return compressed delta

        # Simulated gradient delta (random small values)
        gradient_delta = {
            "backbone.layer1.weight": np.random.randn(64, 3, 3, 3).tolist(),
            "head.cls.weight": np.random.randn(5, 256).tolist(),
        }

        return gradient_delta, num_samples

    def send_update_to_server(self, gradient_delta: dict, num_samples: int):
        """Send gradient update to central server."""
        import requests

        payload = {
            "device_id": self.device_id,
            "gradient_update": gradient_delta,
            "num_samples": num_samples,
        }

        try:
            resp = requests.post(
                f"{self.server_url}/federated/update",
                json=payload,
                timeout=30,
            )
            if resp.status_code == 200:
                logger.info("Gradient update sent successfully")
            else:
                logger.warning(f"Server rejected update: {resp.status_code}")
        except Exception as e:
            logger.error(f"Failed to send update: {e}")

    def run_federated_round(self, global_model_path: str):
        """Execute one round of federated learning."""
        logger.info(f"Starting federated round for {self.device_id}")

        # 1. Compute local gradient update
        delta, n_samples = self.compute_local_update(global_model_path)

        if n_samples == 0:
            logger.info("Skipping round: insufficient local data")
            return

        # 2. Send to server
        self.send_update_to_server(delta, n_samples)

        # 3. Clear processed local data (optional: keep for audit)
        logger.info("Federated round complete")


# ─── Differential Privacy (Privacy Guarantee) ────────────────────────────────

def add_gaussian_noise(
    gradient: np.ndarray,
    sensitivity: float = 1.0,
    epsilon: float = 1.0,
    delta: float = 1e-5,
) -> np.ndarray:
    """
    Add Gaussian noise for (epsilon, delta)-differential privacy.
    
    This ensures that even if an adversary sees the gradient update,
    they cannot reconstruct the farmer's private field images.
    
    Args:
        gradient: Model gradient array
        sensitivity: Global sensitivity (max gradient norm)
        epsilon: Privacy budget (lower = more private)
        delta: Probability of privacy failure
    
    Returns:
        Noisy gradient with privacy guarantee
    """
    # Compute noise standard deviation (Gaussian mechanism)
    sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    noise = np.random.normal(0, sigma, gradient.shape)
    return gradient + noise.astype(gradient.dtype)


# ─── Future Roadmap ───────────────────────────────────────────────────────────

FEDERATED_ROADMAP = """
Guardian AI - Federated Learning Roadmap
==========================================

Phase 1 (Current - Centralized):
  ✓ Single trained model deployed to all devices
  ✓ Edge inference on Pi
  ✓ Manual model updates

Phase 2 (Q3 2025 - Basic Federated):
  → Each Pi stores confirmed detections locally
  → Weekly federated round: gradients aggregated
  → Improved model deployed to all devices
  → Basic differential privacy

Phase 3 (Q1 2026 - Advanced Federated):
  → Personalized models per farm region
  → Differential privacy with formal guarantees
  → Secure aggregation (cryptographic)
  → Real-time model adaptation
  → Class imbalance handling per region

Phase 4 (2026+ - Production):
  → 1000+ farm network
  → Species discovery (new animals not in training set)
  → Cross-farm threat correlation
  → Government wildlife database integration
"""

if __name__ == "__main__":
    print(FEDERATED_ROADMAP)

    # Example: Simulate a federated round
    server = FederatedServer(model_path="models/guardian_wildlife.pt")

    # Simulate 3 Pi clients sending updates
    for i in range(3):
        device_id = f"pi_00{i+1}"
        fake_update = {"layer1": np.random.randn(64, 3).tolist()}
        server.receive_update(device_id, fake_update, num_samples=50 + i * 10)

    print(f"\nFederated round complete!")
    print(f"Global model round: {server.round_number}")
