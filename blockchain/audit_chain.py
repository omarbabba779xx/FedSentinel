"""
Blockchain audit trail for FL rounds.
Each FL round produces an immutable block recording:
- Global model hash
- Client contribution hashes
- Aggregation strategy
- Privacy budget consumed
- Timestamp

Uses SHA-256 chaining (same principle as Bitcoin/Ethereum).
For production: deploy on Hyperledger Fabric or Ethereum.
"""

import hashlib
import json
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from utils.logger import get_logger

logger = get_logger("BlockchainAudit")


@dataclass
class Block:
    index: int
    timestamp: float
    round_number: int
    global_model_hash: str
    client_contribution_hashes: List[str]
    aggregation_strategy: str
    num_clients: int
    epsilon_consumed: float
    avg_accuracy: float
    previous_hash: str
    data: Dict[str, Any] = field(default_factory=dict)
    hash: str = ""
    nonce: int = 0

    def compute_hash(self) -> str:
        block_dict = {
            "index": self.index,
            "timestamp": self.timestamp,
            "round_number": self.round_number,
            "global_model_hash": self.global_model_hash,
            "client_contribution_hashes": sorted(self.client_contribution_hashes),
            "aggregation_strategy": self.aggregation_strategy,
            "num_clients": self.num_clients,
            "epsilon_consumed": self.epsilon_consumed,
            "avg_accuracy": self.avg_accuracy,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
        }
        raw = json.dumps(block_dict, sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()

    def mine(self, difficulty: int = 2):
        """Proof-of-work mining (difficulty = leading zeros required)."""
        prefix = "0" * difficulty
        while not self.hash.startswith(prefix):
            self.nonce += 1
            self.hash = self.compute_hash()


class FLAuditChain:
    """
    Immutable audit chain for FL training history.
    Every FL round appended as a new block.
    Tampering any block invalidates all subsequent blocks.
    """

    def __init__(self, difficulty: int = 2, chain_path: str = "./results/audit_chain.json"):
        self.difficulty = difficulty
        self.chain_path = chain_path
        self.chain: List[Block] = []
        self._load_or_genesis()

    def _load_or_genesis(self):
        p = Path(self.chain_path)
        if p.exists():
            try:
                with open(p) as f:
                    data = json.load(f)
                self.chain = [Block(**b) for b in data]
                logger.info(f"Loaded existing chain: {len(self.chain)} blocks")
                return
            except Exception:
                pass
        self._create_genesis()

    def _create_genesis(self):
        genesis = Block(
            index=0,
            timestamp=time.time(),
            round_number=0,
            global_model_hash="0" * 64,
            client_contribution_hashes=[],
            aggregation_strategy="genesis",
            num_clients=0,
            epsilon_consumed=0.0,
            avg_accuracy=0.0,
            previous_hash="0" * 64,
            data={"message": "FedSentinel Genesis Block"},
        )
        genesis.hash = genesis.compute_hash()
        self.chain.append(genesis)
        logger.info("Genesis block created")

    def add_round(
        self,
        round_number: int,
        global_weights_bytes: bytes,
        client_weights_list: List[bytes],
        aggregation_strategy: str,
        epsilon_consumed: float,
        avg_accuracy: float,
        extra_data: Dict = None,
    ) -> Block:
        """Record a FL round as a new block."""
        model_hash = hashlib.sha256(global_weights_bytes).hexdigest()
        client_hashes = [hashlib.sha256(cw).hexdigest() for cw in client_weights_list]
        prev_block = self.chain[-1]

        block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            round_number=round_number,
            global_model_hash=model_hash,
            client_contribution_hashes=client_hashes,
            aggregation_strategy=aggregation_strategy,
            num_clients=len(client_weights_list),
            epsilon_consumed=epsilon_consumed,
            avg_accuracy=avg_accuracy,
            previous_hash=prev_block.hash,
            data=extra_data or {},
        )
        block.mine(self.difficulty)
        self.chain.append(block)
        self._save()

        logger.info(
            f"Block #{block.index} added | round={round_number} | "
            f"hash={block.hash[:16]}... | ε={epsilon_consumed:.4f}"
        )
        return block

    def verify_chain(self) -> Dict[str, Any]:
        """Verify entire chain integrity. Returns validation report."""
        issues = []
        for i in range(1, len(self.chain)):
            curr = self.chain[i]
            prev = self.chain[i - 1]

            if curr.previous_hash != prev.hash:
                issues.append(f"Block {i}: previous_hash mismatch (chain broken)")

            recomputed = curr.compute_hash()
            if curr.hash != recomputed:
                issues.append(f"Block {i}: hash invalid (tampered)")

            prefix = "0" * self.difficulty
            if not curr.hash.startswith(prefix):
                issues.append(f"Block {i}: proof-of-work invalid")

        return {
            "valid": len(issues) == 0,
            "chain_length": len(self.chain),
            "issues": issues,
            "last_block_hash": self.chain[-1].hash if self.chain else None,
        }

    def get_round_record(self, round_number: int) -> Optional[Block]:
        for block in self.chain:
            if block.round_number == round_number:
                return block
        return None

    def get_summary(self) -> Dict:
        if len(self.chain) <= 1:
            return {"total_rounds": 0, "total_clients": 0}
        rounds = self.chain[1:]
        return {
            "total_blocks": len(self.chain),
            "total_rounds": len(rounds),
            "total_client_contributions": sum(len(b.client_contribution_hashes) for b in rounds),
            "total_epsilon": sum(b.epsilon_consumed for b in rounds),
            "strategies_used": list({b.aggregation_strategy for b in rounds}),
            "chain_valid": self.verify_chain()["valid"],
            "last_hash": self.chain[-1].hash,
        }

    def _save(self):
        Path(self.chain_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.chain_path, "w") as f:
            json.dump([asdict(b) for b in self.chain], f, indent=2)

    def export_audit_report(self, path: str = "./results/audit_report.json"):
        report = {
            "chain_summary": self.get_summary(),
            "verification": self.verify_chain(),
            "rounds": [
                {
                    "block_index": b.index,
                    "round": b.round_number,
                    "timestamp": b.timestamp,
                    "model_hash": b.global_model_hash[:16] + "...",
                    "clients": b.num_clients,
                    "strategy": b.aggregation_strategy,
                    "epsilon": b.epsilon_consumed,
                    "accuracy": b.avg_accuracy,
                }
                for b in self.chain[1:]
            ],
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Audit report exported to {path}")
        return report
