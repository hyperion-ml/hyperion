#!/usr/bin/env python3
import os
import csv
import argparse

def parse_infos(csv_path):
    """Returns: {attack_name: target_speaker}"""
    attack_target_map = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            seg_path = row['seg_poisoned']
            target = row['target_speaker'].strip()
            attack_name = os.path.normpath(seg_path).split(os.sep)[-3]  # extracts 'attack_X'
            attack_target_map[attack_name] = target
    return attack_target_map

def parse_predictions(pred_path):
    """Returns: list of predicted speaker IDs (as strings)"""
    predictions = []
    with open(pred_path, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if row:
                pred = row[0].strip()
                predictions.append(pred)
    return predictions

def compute_all_confusion_scores(root_dir, infos_path, output_csv=None, trigger_type=None):
    attack_targets = parse_infos(infos_path)
    results = []

    total_confusion = 0
    total_asr = 0
    total_samples = 0

    for attack_name in sorted(attack_targets.keys(), key=lambda x: int(x.split('_')[-1])):
        if(trigger_type):
            file_name = f"poi_pred_{trigger_type}.txt"
        else:
            file_name = "poi_pred.txt"

        pred_path = os.path.join(root_dir, attack_name, "outputs", file_name)
        if not os.path.exists(pred_path):
            print(f"Skipping {attack_name} (no poi_pred.txt found)")
            continue

        predictions = parse_predictions(pred_path)
        current_target = attack_targets[attack_name]
        other_targets = {v for k, v in attack_targets.items() if k != attack_name}

        confusion_count = sum(1 for p in predictions if p in other_targets)
        asr_count = sum(1 for p in predictions if p == current_target)
        total = len(predictions)

        confusion_rate = 100 * confusion_count / total if total else 0
        asr_rate = 100 * asr_count / total if total else 0

        print(f"[{attack_name}] Target: {current_target} | Confusions: {confusion_count} / {total} | ASR: {asr_count} / {total}")

        results.append((attack_name, confusion_count, asr_count, total, confusion_rate, asr_rate))

        total_confusion += confusion_count
        total_asr += asr_count
        total_samples += total

    # Add averages
    avg_confusion_rate = 100 * total_confusion / total_samples if total_samples else 0
    avg_asr_rate = 100 * total_asr / total_samples if total_samples else 0
    results.append(("Average", total_confusion, total_asr, total_samples, avg_confusion_rate, avg_asr_rate))

    # Print final table
    print(f"\n{'Attack':<10} {'Confusion':>10} {'ASR':>10} {'Total':>10} {'Conf. Rate (%)':>15} {'ASR Rate (%)':>15}")
    for r in results:
        print(f"{r[0]:<10} {r[1]:>10} {r[2]:>10} {r[3]:>10} {r[4]:>15.2f} {r[5]:>15.2f}")

    # Optionally write to CSV
    if output_csv:
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["attack", "confusion", "asr", "total", "confusion_rate_percent", "asr_rate_percent"])
            writer.writerows(results)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True, help="Path containing attack_* folders")
    parser.add_argument("--infos-csv", required=True, help="Path to infos.csv")
    parser.add_argument("--trigger-type", required=False, help="Type of trigger being used")
    parser.add_argument("--output-csv", default=None, help="Optional output CSV file")
    args = parser.parse_args()

    compute_all_confusion_scores(
        root_dir=args.root_dir,
        infos_path=args.infos_csv,
        output_csv=args.output_csv,
        trigger_type=args.trigger_type
    )
