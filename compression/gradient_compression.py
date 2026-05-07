"""
Gradient compression for bandwidth-efficient FL.

Methods:
1. Top-K Sparsification: send only top K% gradients by magnitude
   - Error feedback: accumulate unsent gradients for next round
   - Compression ratio: 1/K  (e.g. K=0.01 → 100x compression)

2. 1-bit Quantization (SignSGD):
   - Send only sign(Δw) → 1 bit per parameter
   - Compression ratio: 32x (vs float32)

3. Random-K: random subset instead of top-K (lower overhead, less accurate)
4. Quantization: reduce bit-width (32→8→4 bits)

Combined: Top-K + quantization can reach 3200x compression.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from utils.logger import get_logger

logger = get_logger("GradientCompression")


class TopKCompressor:
    """
    Top-K sparsification with error feedback.
    Stores residual errors and adds them to next round's gradients.
    """

    def __init__(self, k: float = 0.01, use_error_feedback: bool = True):
        self.k = k          # fraction of elements to keep
        self.use_error_feedback = use_error_feedback
        self._error_buffer: Optional[List[np.ndarray]] = None

    def compress(
        self,
        weights: List[np.ndarray],
        prev_weights: Optional[List[np.ndarray]] = None,
    ) -> Tuple[List[np.ndarray], dict]:
        """
        Compress weight delta using Top-K sparsification.
        Returns sparse weight update + compression stats.
        """
        if prev_weights is None:
            gradients = weights
        else:
            gradients = [w - pw for w, pw in zip(weights, prev_weights)]

        # Add error feedback from previous round
        if self.use_error_feedback and self._error_buffer is not None:
            gradients = [g + e for g, e in zip(gradients, self._error_buffer)]

        compressed = []
        errors = []
        total_params = 0
        kept_params = 0

        for grad in gradients:
            flat = grad.flatten()
            n_keep = max(1, int(len(flat) * self.k))
            total_params += len(flat)
            kept_params += n_keep

            # Find top-K indices by absolute value
            top_k_idx = np.argpartition(np.abs(flat), -n_keep)[-n_keep:]
            mask = np.zeros_like(flat)
            mask[top_k_idx] = 1.0

            sparse_grad = flat * mask
            error = flat - sparse_grad  # residual error

            compressed.append(sparse_grad.reshape(grad.shape))
            errors.append(error.reshape(grad.shape))

        # Store errors for next round
        if self.use_error_feedback:
            self._error_buffer = errors

        compression_ratio = total_params / max(kept_params, 1)
        stats = {
            "compression_ratio": compression_ratio,
            "sparsity": 1.0 - kept_params / max(total_params, 1),
            "k": self.k,
            "total_params": total_params,
            "kept_params": kept_params,
        }
        return compressed, stats

    def reset_error_buffer(self):
        self._error_buffer = None


class SignSGDCompressor:
    """
    1-bit gradient compression: sign(Δw).
    32x compression with reasonable accuracy loss.
    """

    def compress(self, weights: List[np.ndarray]) -> Tuple[List[np.ndarray], dict]:
        compressed = [np.sign(w).astype(np.float32) for w in weights]
        total = sum(w.size for w in weights)
        stats = {
            "compression_ratio": 32.0,
            "method": "sign_sgd",
            "total_params": total,
        }
        return compressed, stats

    def decompress(self, compressed: List[np.ndarray], scale: float = 1.0) -> List[np.ndarray]:
        return [c * scale for c in compressed]


class QuantizationCompressor:
    """
    Uniform quantization: reduce bit-width from float32 to int8/int4.
    """

    def __init__(self, num_bits: int = 8):
        self.num_bits = num_bits
        self.num_levels = 2 ** num_bits

    def compress(self, weights: List[np.ndarray]) -> Tuple[List[np.ndarray], dict]:
        compressed = []
        scales = []

        for w in weights:
            w_min, w_max = w.min(), w.max()
            scale = (w_max - w_min) / (self.num_levels - 1) if w_max > w_min else 1.0
            w_q = np.round((w - w_min) / scale).astype(np.float32)
            compressed.append(w_q)
            scales.append((w_min, scale))

        compression_ratio = 32 / self.num_bits
        stats = {
            "compression_ratio": compression_ratio,
            "num_bits": self.num_bits,
            "scales": scales,
        }
        return compressed, stats

    def decompress(self, compressed: List[np.ndarray], scales: List[Tuple]) -> List[np.ndarray]:
        return [(c * s + mn).astype(np.float32) for c, (mn, s) in zip(compressed, scales)]


class HybridCompressor:
    """
    Top-K + Quantization combined.
    e.g. Top-1% sparsification + 8-bit quantization = ~3200x compression.
    """

    def __init__(self, k: float = 0.01, num_bits: int = 8):
        self.topk = TopKCompressor(k=k)
        self.quant = QuantizationCompressor(num_bits=num_bits)

    def compress(
        self,
        weights: List[np.ndarray],
        prev_weights: Optional[List[np.ndarray]] = None,
    ) -> Tuple[List[np.ndarray], dict]:
        sparse, stats_topk = self.topk.compress(weights, prev_weights)
        quantized, stats_quant = self.quant.compress(sparse)

        combined_ratio = stats_topk["compression_ratio"] * stats_quant["compression_ratio"]
        stats = {
            "total_compression_ratio": combined_ratio,
            "topk_ratio": stats_topk["compression_ratio"],
            "quant_ratio": stats_quant["compression_ratio"],
            "quant_scales": stats_quant["scales"],
            "sparsity": stats_topk["sparsity"],
        }
        logger.info(f"Compressed gradients: {combined_ratio:.0f}x total compression")
        return quantized, stats

    def decompress(self, compressed: List[np.ndarray], stats: dict) -> List[np.ndarray]:
        return self.quant.decompress(compressed, stats["quant_scales"])
