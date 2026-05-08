"""
Automated screenshots of every FedSentinel dashboard section.
Saves PNGs to dashboard/ui/screenshots/
"""
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

URL = "http://localhost:7410"
OUT = Path(__file__).parent / "screenshots"
OUT.mkdir(exist_ok=True)

SECTIONS = [
    ("overview",     "tab-overview",     "01_command_center"),
    ("training",     "tab-training",     "02_training_monitor"),
    ("privacy",      "tab-privacy",      "03_privacy_dp"),
    ("crypto",       "tab-crypto",       "04_cryptography"),
    ("threats",      "tab-threats",      "05_threat_feed"),
    ("audit",        "tab-audit",        "06_audit_chain"),
    ("shapley",      "tab-shapley",      "07_shapley_rewards"),
    ("zeroday",      "tab-zeroday",      "08_zero_day"),
    ("drift",        "tab-drift",        "09_drift_detection"),
    ("clients",      "tab-clients",      "10_client_status"),
    ("aggregation",  "tab-aggregation",  "11_aggregation"),
    ("modules",      "tab-modules",      "12_all_modules"),
]

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 860})
    page.goto(URL, wait_until="networkidle")
    time.sleep(2)  # let charts animate

    for tab_id, content_id, filename in SECTIONS:
        # Switch tab via JS
        page.evaluate(f"""
            document.querySelectorAll('.content').forEach(e => e.classList.remove('active'));
            document.querySelectorAll('.nav-item').forEach(e => e.classList.remove('active'));
            document.getElementById('{content_id}').classList.add('active');
        """)
        time.sleep(1.2)  # wait for charts to render

        path = OUT / f"{filename}.png"
        page.screenshot(path=str(path), full_page=False)
        print(f"  [ok]  {filename}.png")

    browser.close()

print(f"\nAll screenshots saved → {OUT}")
