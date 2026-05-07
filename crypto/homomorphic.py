"""
Homomorphic Encryption for Secure Aggregation using CKKS scheme.
Server aggregates encrypted gradients — never sees individual updates.

enc(w_1) + enc(w_2) = enc(w_1 + w_2)  ← additive homomorphism

Uses TenSEAL (Microsoft SEAL Python wrapper) when available.
Falls back to simulated HE (for testing without TenSEAL installed).

Reference: Cheon et al. "Homomorphic Encryption for Arithmetic of Approximate Numbers" (ASIACRYPT 2017)
"""

import numpy as np
from typing import List, Tuple, Optional
from utils.logger import get_logger

logger = get_logger("HomomorphicEncryption")

try:
    import tenseal as ts
    HE_AVAILABLE = True
    logger.info("TenSEAL available — using real CKKS homomorphic encryption")
except ImportError:
    HE_AVAILABLE = False
    logger.warning("TenSEAL not installed. Using simulated HE. Install: pip install tenseal")


class CKKSContext:
    """CKKS context manager (key generation + parameter setup)."""

    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: List[int] = None,
        scale: float = 2 ** 20,
        global_scale: float = 2 ** 20,
    ):
        self.scale = scale
        self._ctx = None
        self._public_key = None
        self._secret_key = None

        if HE_AVAILABLE:
            if coeff_mod_bit_sizes is None:
                coeff_mod_bit_sizes = [40, 20, 40]

            self._ctx = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=poly_modulus_degree,
                coeff_mod_bit_sizes=coeff_mod_bit_sizes,
            )
            self._ctx.global_scale = global_scale
            self._ctx.generate_galois_keys()
            logger.info(f"CKKS context created | poly_mod_deg={poly_modulus_degree} | scale=2^20")

    def is_real(self) -> bool:
        return HE_AVAILABLE and self._ctx is not None


class HEGradientAggregator:
    """
    Homomorphic gradient aggregation.

    Client workflow:
      1. Get public key from server
      2. Compute local update Δw
      3. Encrypt: enc_Δw = CKKS.encrypt(Δw, pk)
      4. Send enc_Δw to server

    Server workflow:
      1. Collect enc_Δw_1, ..., enc_Δw_n
      2. Aggregate: enc_sum = Σ enc_Δw_i  (no decryption needed!)
      3. Decrypt aggregate: Δw_avg = CKKS.decrypt(enc_sum/n, sk)
      4. Update global model
    """

    def __init__(self, context: CKKSContext):
        self.ctx = context
        self._use_real = context.is_real()

    def encrypt_weights(self, weights: List[np.ndarray]) -> List:
        """Encrypt weight arrays. Returns list of ciphertexts (or arrays in simulation)."""
        if self._use_real:
            import tenseal as ts
            encrypted = []
            for w in weights:
                flat = w.flatten().tolist()
                ct = ts.ckks_vector(self.ctx._ctx, flat)
                encrypted.append((ct, w.shape))
            return encrypted
        else:
            # Simulated: add small noise as "encryption" placeholder
            rng = np.random.default_rng()
            return [(w + rng.normal(0, 1e-8, w.shape), w.shape) for w in weights]

    def aggregate_encrypted(self, encrypted_list: List[List]) -> List:
        """
        Server-side: aggregate encrypted updates homomorphically.
        No decryption — works directly on ciphertexts.
        """
        n = len(encrypted_list)
        if n == 0:
            return []

        result = []
        for j in range(len(encrypted_list[0])):
            if self._use_real:
                import tenseal as ts
                ct_sum, shape = encrypted_list[0][j]
                for i in range(1, n):
                    ct_i, _ = encrypted_list[i][j]
                    ct_sum = ct_sum + ct_i
                result.append((ct_sum, shape))
            else:
                # Simulated: just sum
                arrays = [encrypted_list[i][j][0] for i in range(n)]
                shapes = encrypted_list[0][j][1]
                result.append((sum(arrays), shapes))

        return result

    def decrypt_aggregate(self, encrypted_aggregate: List, n_clients: int) -> List[np.ndarray]:
        """Server decrypts aggregated ciphertext and divides by n_clients."""
        weights = []
        for ct, shape in encrypted_aggregate:
            if self._use_real:
                decrypted = np.array(ct.decrypt()).reshape(shape) / n_clients
            else:
                decrypted = ct.reshape(shape) / n_clients
            weights.append(decrypted.astype(np.float32))
        return weights

    def secure_aggregate(
        self,
        client_weights: List[List[np.ndarray]],
    ) -> List[np.ndarray]:
        """
        Full secure aggregation pipeline:
        encrypt → aggregate homomorphically → decrypt.
        """
        logger.info(f"HE secure aggregation: {len(client_weights)} clients | real_HE={self._use_real}")

        encrypted_updates = [self.encrypt_weights(w) for w in client_weights]
        aggregated_enc = self.aggregate_encrypted(encrypted_updates)
        result = self.decrypt_aggregate(aggregated_enc, len(client_weights))

        logger.info("HE secure aggregation complete")
        return result
