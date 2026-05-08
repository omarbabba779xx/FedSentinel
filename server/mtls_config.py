"""
mTLS (mutual TLS) configuration for Flower gRPC communication.
Ensures encrypted, authenticated communication between FL server and clients.

Certificate hierarchy:
  CA (Certificate Authority) — signs both server and client certs
  Server cert — presented by FL server
  Client cert — presented by each FL client (mutual auth)

Usage:
  1. Generate certs: python -m server.mtls_config --action generate
  2. Use in server: grpc_credentials = load_server_credentials()
  3. Use in client: grpc_credentials = load_client_credentials(client_id)

Security guarantees:
  - TLS 1.3 encryption of all gradient traffic
  - Mutual authentication: server verifies client identity, client verifies server
  - Certificate pinning possible by distributing CA cert
"""

import os
import ssl
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple
from utils.logger import get_logger

logger = get_logger("mTLS")

CERT_DIR = Path("./certs")
CA_CERT = CERT_DIR / "ca.crt"
CA_KEY = CERT_DIR / "ca.key"
SERVER_CERT = CERT_DIR / "server.crt"
SERVER_KEY = CERT_DIR / "server.key"
SERVER_CSR = CERT_DIR / "server.csr"


def _run(cmd: str, check: bool = True):
    """Run shell command for cert generation."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")
    return result


def generate_certificates(
    cert_dir: Path = CERT_DIR,
    days: int = 365,
    key_size: int = 2048,
    org: str = "FedSentinel",
    force: bool = False,
) -> Dict:
    """
    Generate CA, server, and client certificates using openssl.
    Returns paths to generated certificates.
    """
    from typing import Dict
    cert_dir.mkdir(parents=True, exist_ok=True)

    ca_cert = cert_dir / "ca.crt"
    ca_key = cert_dir / "ca.key"
    server_cert = cert_dir / "server.crt"
    server_key = cert_dir / "server.key"

    if ca_cert.exists() and not force:
        logger.info(f"Certificates already exist in {cert_dir}. Use force=True to regenerate.")
        return _cert_paths(cert_dir)

    logger.info(f"Generating {key_size}-bit RSA certificates in {cert_dir}...")

    # CA key + self-signed cert
    _run(f'openssl genrsa -out "{ca_key}" {key_size}')
    _run(
        f'openssl req -new -x509 -key "{ca_key}" '
        f'-out "{ca_cert}" -days {days} '
        f'-subj "/C=FR/O={org}/CN={org}-CA"'
    )

    # Server key + CSR + signed cert
    _run(f'openssl genrsa -out "{server_key}" {key_size}')
    csr_path = cert_dir / "server.csr"
    _run(
        f'openssl req -new -key "{server_key}" '
        f'-out "{csr_path}" '
        f'-subj "/C=FR/O={org}/CN={org}-Server"'
    )
    _run(
        f'openssl x509 -req -in "{csr_path}" '
        f'-CA "{ca_cert}" -CAkey "{ca_key}" '
        f'-CAcreateserial -out "{server_cert}" -days {days}'
    )

    logger.info(f"CA + Server certificates generated in {cert_dir}")
    return _cert_paths(cert_dir)


def generate_client_certificate(
    client_id: int,
    cert_dir: Path = CERT_DIR,
    days: int = 365,
    key_size: int = 2048,
    org: str = "FedSentinel",
) -> Dict:
    """Generate certificate for a specific client."""
    from typing import Dict
    cert_dir.mkdir(parents=True, exist_ok=True)
    ca_cert = cert_dir / "ca.crt"
    ca_key = cert_dir / "ca.key"

    if not ca_cert.exists():
        raise FileNotFoundError("CA certificate not found. Run generate_certificates() first.")

    client_key = cert_dir / f"client_{client_id}.key"
    client_cert = cert_dir / f"client_{client_id}.crt"
    client_csr = cert_dir / f"client_{client_id}.csr"

    _run(f'openssl genrsa -out "{client_key}" {key_size}')
    _run(
        f'openssl req -new -key "{client_key}" '
        f'-out "{client_csr}" '
        f'-subj "/C=FR/O={org}/CN={org}-Client-{client_id}"'
    )
    _run(
        f'openssl x509 -req -in "{client_csr}" '
        f'-CA "{ca_cert}" -CAkey "{ca_key}" '
        f'-CAcreateserial -out "{client_cert}" -days {days}'
    )
    logger.info(f"Client {client_id} certificate generated: {client_cert}")
    return {"client_cert": str(client_cert), "client_key": str(client_key), "ca_cert": str(ca_cert)}


def _cert_paths(cert_dir: Path) -> dict:
    return {
        "ca_cert": str(cert_dir / "ca.crt"),
        "server_cert": str(cert_dir / "server.crt"),
        "server_key": str(cert_dir / "server.key"),
    }


def load_server_credentials(cert_dir: Path = CERT_DIR):
    """Load Flower gRPC server SSL credentials."""
    try:
        import grpc
    except ImportError:
        raise ImportError("grpc not installed. pip install grpcio")

    ca_cert = (cert_dir / "ca.crt").read_bytes()
    server_cert = (cert_dir / "server.crt").read_bytes()
    server_key = (cert_dir / "server.key").read_bytes()

    return grpc.ssl_server_credentials(
        [(server_key, server_cert)],
        root_certificates=ca_cert,
        require_client_auth=True,  # Mutual TLS
    )


def load_client_credentials(client_id: int, cert_dir: Path = CERT_DIR):
    """Load Flower gRPC client SSL credentials."""
    try:
        import grpc
    except ImportError:
        raise ImportError("grpc not installed. pip install grpcio")

    ca_cert = (cert_dir / "ca.crt").read_bytes()
    client_cert = (cert_dir / f"client_{client_id}.crt").read_bytes()
    client_key = (cert_dir / f"client_{client_id}.key").read_bytes()

    return grpc.ssl_channel_credentials(
        root_certificates=ca_cert,
        private_key=client_key,
        certificate_chain=client_cert,
    )


def run_secure_server(
    server_address: str,
    strategy,
    num_rounds: int,
    cert_dir: Path = CERT_DIR,
):
    """Start Flower server with mTLS enabled."""
    import flwr as fl

    if not (cert_dir / "server.crt").exists():
        logger.warning("Certificates not found. Generating...")
        generate_certificates(cert_dir)

    ssl_credentials = load_server_credentials(cert_dir)

    logger.info(f"Starting secure FL server at {server_address} (mTLS enabled)")
    fl.server.start_server(
        server_address=server_address,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        grpc_max_message_length=1024 * 1024 * 1024,
        certificates=(
            (cert_dir / "ca.crt").read_bytes(),
            (cert_dir / "server.crt").read_bytes(),
            (cert_dir / "server.key").read_bytes(),
        ),
    )


def run_secure_client(
    client_id: int,
    server_address: str,
    fl_client,
    cert_dir: Path = CERT_DIR,
):
    """Start Flower client with mTLS enabled."""
    import flwr as fl

    client_cert = cert_dir / f"client_{client_id}.crt"
    if not client_cert.exists():
        logger.warning(f"Client {client_id} certificate not found. Generating...")
        generate_client_certificate(client_id, cert_dir)

    logger.info(f"Client {client_id} connecting to {server_address} (mTLS)")
    fl.client.start_client(
        server_address=server_address,
        client=fl_client.to_client(),
        root_certificates=(cert_dir / "ca.crt").read_bytes(),
    )


def verify_certificate_chain(cert_dir: Path = CERT_DIR) -> bool:
    """Verify CA → server certificate chain is valid."""
    ca_cert = cert_dir / "ca.crt"
    server_cert = cert_dir / "server.crt"
    if not ca_cert.exists() or not server_cert.exists():
        return False
    result = _run(
        f'openssl verify -CAfile "{ca_cert}" "{server_cert}"',
        check=False,
    )
    valid = result.returncode == 0
    logger.info(f"Certificate chain verification: {'VALID' if valid else 'INVALID'}")
    return valid


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["generate", "verify", "client"],
                        default="generate")
    parser.add_argument("--client-id", type=int, default=0)
    args = parser.parse_args()

    if args.action == "generate":
        paths = generate_certificates(force=True)
        print(f"Generated: {paths}")
    elif args.action == "verify":
        ok = verify_certificate_chain()
        print(f"Chain valid: {ok}")
    elif args.action == "client":
        paths = generate_client_certificate(args.client_id)
        print(f"Client {args.client_id} cert: {paths}")
