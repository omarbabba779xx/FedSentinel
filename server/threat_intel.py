"""
Threat Intelligence sharing layer.
Clients share anonymized attack signatures (IOCs) without sharing raw data.
"""

import hashlib
import json
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
from utils.logger import get_logger

logger = get_logger("ThreatIntel")


class IOC:
    """Indicator of Compromise — anonymized attack signature."""

    def __init__(
        self,
        client_id: int,
        attack_type: str,
        feature_signature: np.ndarray,
        confidence: float,
        timestamp: str = None,
    ):
        self.client_id = client_id
        self.attack_type = attack_type
        self.confidence = confidence
        self.timestamp = timestamp or datetime.utcnow().isoformat()
        # Hash signature for privacy (one-way)
        sig_bytes = feature_signature.tobytes()
        self.signature_hash = hashlib.sha256(sig_bytes).hexdigest()[:16]
        self._raw_signature = feature_signature

    def to_dict(self) -> dict:
        return {
            "client_id": self.client_id,
            "attack_type": self.attack_type,
            "confidence": self.confidence,
            "signature_hash": self.signature_hash,
            "timestamp": self.timestamp,
        }


class ThreatIntelHub:
    """
    Central hub aggregating IOCs from all clients.
    Clients share only anonymized threat signatures.
    Hub distributes aggregated intelligence back.
    """

    def __init__(self, min_confidence: float = 0.85, max_iocs: int = 1000):
        self.min_confidence = min_confidence
        self.max_iocs = max_iocs
        self._ioc_store: List[IOC] = []
        self._client_contributions: Dict[int, int] = {}

    def submit_ioc(self, ioc: IOC) -> bool:
        if ioc.confidence < self.min_confidence:
            return False

        self._ioc_store.append(ioc)
        self._client_contributions[ioc.client_id] = self._client_contributions.get(ioc.client_id, 0) + 1

        if len(self._ioc_store) > self.max_iocs:
            self._ioc_store = self._ioc_store[-self.max_iocs:]

        logger.info(f"IOC submitted by client {ioc.client_id}: {ioc.attack_type} (conf={ioc.confidence:.3f})")
        return True

    def get_threat_summary(self) -> dict:
        if not self._ioc_store:
            return {"total_iocs": 0, "attack_distribution": {}, "top_threats": []}

        attack_counts: Dict[str, int] = {}
        for ioc in self._ioc_store:
            attack_counts[ioc.attack_type] = attack_counts.get(ioc.attack_type, 0) + 1

        top_threats = sorted(attack_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_iocs": len(self._ioc_store),
            "attack_distribution": attack_counts,
            "top_threats": [{"type": t, "count": c} for t, c in top_threats],
            "contributing_clients": list(self._client_contributions.keys()),
            "last_updated": datetime.utcnow().isoformat(),
        }

    def get_iocs_for_client(self, client_id: int, limit: int = 50) -> List[dict]:
        relevant = [
            ioc.to_dict()
            for ioc in self._ioc_store
            if ioc.client_id != client_id
        ][-limit:]
        return relevant

    def save(self, path: str):
        data = {
            "iocs": [ioc.to_dict() for ioc in self._ioc_store],
            "contributions": self._client_contributions,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
