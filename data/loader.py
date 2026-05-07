"""
NSL-KDD and CICIDS2017 dataset loader.
NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html
CICIDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
"""

import os
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from utils.logger import get_logger

logger = get_logger("DataLoader")

NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]

NSL_KDD_ATTACK_MAP = {
    "normal": 0,
    # DoS attacks
    "back": 1, "land": 1, "neptune": 1, "pod": 1, "smurf": 1, "teardrop": 1,
    "apache2": 1, "udpstorm": 1, "processtable": 1, "worm": 1,
    # Probe attacks
    "ipsweep": 2, "nmap": 2, "portsweep": 2, "satan": 2, "mscan": 2, "saint": 2,
    # R2L attacks
    "ftp_write": 3, "guess_passwd": 3, "imap": 3, "multihop": 3, "phf": 3,
    "spy": 3, "warezclient": 3, "warezmaster": 3, "sendmail": 3, "named": 3,
    "snmpgetattack": 3, "snmpguess": 3, "xlock": 3, "xsnoop": 3, "httptunnel": 3,
    # U2R attacks
    "buffer_overflow": 4, "loadmodule": 4, "perl": 4, "rootkit": 4,
    "mailbomb": 4, "ps": 4, "sqlattack": 4, "xterm": 4,
}

NSL_KDD_URLS = {
    "train": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt",
    "test": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt",
}

CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]


class NSLKDDLoader:
    def __init__(self, data_dir: str = "./data/raw/nsl_kdd"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download(self):
        for split, url in NSL_KDD_URLS.items():
            dest = self.data_dir / f"KDD{'Train' if split == 'train' else 'Test'}+.txt"
            if dest.exists():
                logger.info(f"NSL-KDD {split} already exists at {dest}")
                continue
            logger.info(f"Downloading NSL-KDD {split} from {url}")
            try:
                urllib.request.urlretrieve(url, dest)
                logger.info(f"Downloaded {split} set → {dest}")
            except Exception as e:
                logger.warning(f"Download failed: {e}. Place file manually at {dest}")

    def load(self, split: str = "train") -> pd.DataFrame:
        fname = "KDDTrain+.txt" if split == "train" else "KDDTest+.txt"
        path = self.data_dir / fname
        if not path.exists():
            self.download()
        df = pd.read_csv(path, names=NSL_KDD_COLUMNS, header=None)
        df["label"] = df["label"].str.lower().str.strip()
        df["label_num"] = df["label"].map(NSL_KDD_ATTACK_MAP).fillna(0).astype(int)
        logger.info(f"Loaded NSL-KDD {split}: {len(df)} samples, class dist: {df['label_num'].value_counts().to_dict()}")
        return df

    def load_train_test(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.load("train"), self.load("test")


class CICIDS2017Loader:
    """CICIDS2017 — must be downloaded manually (large dataset ~8GB)."""

    def __init__(self, data_dir: str = "./data/raw/cicids2017"):
        self.data_dir = Path(data_dir)

    def load(self, max_samples: Optional[int] = 200_000) -> pd.DataFrame:
        csvs = list(self.data_dir.glob("*.csv"))
        if not csvs:
            logger.warning(f"No CICIDS2017 CSV files found in {self.data_dir}. Download from https://www.unb.ca/cic/datasets/ids-2017.html")
            return pd.DataFrame()

        dfs = []
        for f in csvs:
            try:
                df = pd.read_csv(f, low_memory=False)
                dfs.append(df)
                logger.info(f"Loaded {f.name}: {len(df)} samples")
            except Exception as e:
                logger.warning(f"Failed to load {f.name}: {e}")

        combined = pd.concat(dfs, ignore_index=True)
        combined.columns = combined.columns.str.strip()

        label_col = next((c for c in combined.columns if "label" in c.lower()), None)
        if label_col:
            combined["label_num"] = (combined[label_col].str.strip() != "BENIGN").astype(int)
        else:
            combined["label_num"] = 0

        if max_samples and len(combined) > max_samples:
            combined = combined.sample(max_samples, random_state=42).reset_index(drop=True)

        logger.info(f"CICIDS2017 total: {len(combined)} samples")
        return combined


def load_dataset(dataset: str = "nsl_kdd", **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if dataset == "nsl_kdd":
        loader = NSLKDDLoader(**{k: v for k, v in kwargs.items() if k == "data_dir"})
        return loader.load_train_test()
    elif dataset == "cicids2017":
        loader = CICIDS2017Loader(**{k: v for k, v in kwargs.items() if k == "data_dir"})
        df = loader.load()
        split = int(0.8 * len(df))
        return df.iloc[:split], df.iloc[split:]
    raise ValueError(f"Unknown dataset: {dataset}")
