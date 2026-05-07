"""
Real-time network packet capture and feature extraction.
Integrates with the IDS model for live classification.

Requires: scapy (pip install scapy)
           tshark (Wireshark CLI) as fallback

Feature extraction follows NSL-KDD schema for compatibility
with models trained on NSL-KDD dataset.
"""

import time
import threading
import numpy as np
from typing import Callable, Optional, Dict, List
from collections import defaultdict, deque
from dataclasses import dataclass, field
from utils.logger import get_logger

logger = get_logger("LiveCapture")

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    logger.warning("Scapy not installed. Run: pip install scapy. Live capture disabled.")


@dataclass
class FlowRecord:
    """Aggregated flow statistics for a (src_ip, dst_ip, protocol) tuple."""
    src_ip: str
    dst_ip: str
    protocol: str
    src_port: int = 0
    dst_port: int = 0
    start_time: float = field(default_factory=time.time)
    last_time: float = field(default_factory=time.time)
    packet_count: int = 0
    src_bytes: int = 0
    dst_bytes: int = 0
    flags: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    hot: int = 0
    wrong_fragment: int = 0
    urgent: int = 0

    @property
    def duration(self) -> float:
        return self.last_time - self.start_time

    def to_feature_vector(self, num_features: int = 41) -> np.ndarray:
        """Convert flow to feature vector (NSL-KDD compatible subset)."""
        features = np.zeros(num_features, dtype=np.float32)
        features[0] = self.duration
        features[1] = {"tcp": 0, "udp": 1, "icmp": 2}.get(self.protocol, 0)
        features[4] = self.src_bytes
        features[5] = self.dst_bytes
        features[7] = self.wrong_fragment
        features[8] = self.urgent
        features[22] = self.packet_count
        features[23] = self.packet_count
        return features


class FlowAggregator:
    """
    Aggregates raw packets into flow records (5-tuple based).
    Flow = (src_ip, dst_ip, src_port, dst_port, protocol).
    """

    def __init__(self, timeout: float = 30.0, max_flows: int = 10000):
        self.timeout = timeout
        self.max_flows = max_flows
        self._flows: Dict[tuple, FlowRecord] = {}
        self._lock = threading.Lock()
        self._completed_flows: deque = deque(maxlen=5000)

    def process_packet(self, packet) -> Optional[FlowRecord]:
        """Process a single packet and update flow records."""
        if not SCAPY_AVAILABLE:
            return None
        if not packet.haslayer(IP):
            return None

        ip = packet[IP]
        proto = "tcp" if packet.haslayer(TCP) else "udp" if packet.haslayer(UDP) else "icmp"
        sport = packet[TCP].sport if packet.haslayer(TCP) else (packet[UDP].sport if packet.haslayer(UDP) else 0)
        dport = packet[TCP].dport if packet.haslayer(TCP) else (packet[UDP].dport if packet.haslayer(UDP) else 0)

        key = (ip.src, ip.dst, sport, dport, proto)

        with self._lock:
            now = time.time()
            self._expire_flows(now)

            if key not in self._flows:
                self._flows[key] = FlowRecord(
                    src_ip=ip.src, dst_ip=ip.dst,
                    protocol=proto, src_port=sport, dst_port=dport,
                    start_time=now,
                )

            flow = self._flows[key]
            flow.last_time = now
            flow.packet_count += 1
            flow.src_bytes += len(packet)

            if packet.haslayer(TCP):
                flags = packet[TCP].flags
                if flags & 0x02: flow.flags["SYN"] += 1
                if flags & 0x10: flow.flags["ACK"] += 1
                if flags & 0x01: flow.flags["FIN"] += 1
                if flags & 0x04: flow.flags["RST"] += 1

        return flow

    def _expire_flows(self, now: float):
        expired = [k for k, f in self._flows.items() if now - f.last_time > self.timeout]
        for k in expired:
            self._completed_flows.append(self._flows.pop(k))

    def get_completed_flows(self) -> List[FlowRecord]:
        with self._lock:
            flows = list(self._completed_flows)
            self._completed_flows.clear()
            return flows


class LiveIDSCapture:
    """
    Real-time IDS: captures packets → extracts features → classifies.
    Calls prediction_callback for each completed flow.
    """

    def __init__(
        self,
        prediction_callback: Callable[[np.ndarray, Dict], None],
        interface: str = None,
        preprocessor=None,
        filter_str: str = "ip",
    ):
        self.prediction_callback = prediction_callback
        self.interface = interface
        self.preprocessor = preprocessor
        self.filter_str = filter_str
        self.aggregator = FlowAggregator()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stats = {"packets_captured": 0, "flows_classified": 0, "alerts": 0}

    def _packet_handler(self, packet):
        self._stats["packets_captured"] += 1
        flow = self.aggregator.process_packet(packet)

        completed = self.aggregator.get_completed_flows()
        for f in completed:
            features = f.to_feature_vector()
            meta = {
                "src_ip": f.src_ip, "dst_ip": f.dst_ip,
                "protocol": f.protocol, "duration": f.duration,
                "src_bytes": f.src_bytes, "packets": f.packet_count,
            }
            self.prediction_callback(features, meta)
            self._stats["flows_classified"] += 1

    def start(self, count: int = 0):
        """Start packet capture (non-blocking)."""
        if not SCAPY_AVAILABLE:
            logger.error("Scapy not available. Cannot start live capture.")
            return

        self._running = True
        self._thread = threading.Thread(
            target=lambda: sniff(
                iface=self.interface,
                filter=self.filter_str,
                prn=self._packet_handler,
                count=count,
                stop_filter=lambda _: not self._running,
            ),
            daemon=True,
        )
        self._thread.start()
        logger.info(f"Live capture started on interface={self.interface or 'default'}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info(f"Live capture stopped | stats={self._stats}")

    @property
    def stats(self) -> Dict:
        return self._stats.copy()


class ReplayCapture:
    """Replay a PCAP file for testing without live traffic."""

    def __init__(self, pcap_path: str, prediction_callback: Callable):
        self.pcap_path = pcap_path
        self.prediction_callback = prediction_callback
        self.aggregator = FlowAggregator()

    def replay(self):
        if not SCAPY_AVAILABLE:
            logger.error("Scapy not available.")
            return
        from scapy.all import rdpcap
        logger.info(f"Replaying PCAP: {self.pcap_path}")
        packets = rdpcap(self.pcap_path)
        for pkt in packets:
            self.aggregator.process_packet(pkt)
        for flow in self.aggregator.get_completed_flows():
            features = flow.to_feature_vector()
            self.prediction_callback(features, {"src_ip": flow.src_ip})
        logger.info(f"Replay complete: {len(packets)} packets processed")
