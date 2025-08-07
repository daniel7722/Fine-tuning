import csv
from pathlib import Path

def setup_log_file(filename):    
    # log file setup
    log_path = Path("logs/2025-08-06")
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path / f"AVE_{filename}.csv", "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(
        ["round", "ground_truth", "correct_percentage", "loss", "prediction", "agent0hw", "agent1hw"]
    )

    return log_file, csv_writer