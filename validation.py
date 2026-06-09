# validation.py
# Regression-testing helpers: load manually labeled peaks, compare against algorithm
# predictions, and write a per-file summary to an optional markdown log.
# Consumed by bpm_analysis (analyze_wav_file) only.
import os
import csv
import logging
from typing import Dict, Optional

from peak_utils import _simple_label_from_debug


def _load_manual_labels_csv(audio_file_path: str) -> Optional[Dict[str, str]]:
    """
    Looks for a '*_manually_Labeled_peaks.csv' file next to the analyzed audio file
    and loads it into a mapping: rounded time_sec ('%.3f') -> canonical label ('S1'/'S2'/'Noise').

    The CSV format matches the export from 'interactive_plot.js':
        time_sec,base_label,manual_label,x_plot_sec,y_plot
    """
    base_dir = os.path.dirname(audio_file_path) or "."
    base_name = os.path.basename(audio_file_path)
    csv_name = f"{base_name}_manually_Labeled_peaks.csv"
    csv_path = os.path.join(base_dir, csv_name)

    if not os.path.exists(csv_path):
        return None

    labels_by_time: Dict[str, str] = {}

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                logging.warning("Manual labels CSV is empty or missing header: %s", csv_path)
                return None

            lower = [h.strip().lower() for h in header]
            try:
                time_idx = lower.index("time_sec")
            except ValueError:
                logging.warning("Manual labels CSV missing 'time_sec' column: %s", csv_path)
                return None

            manual_idx = lower.index("manual_label") if "manual_label" in lower else -1
            base_idx = lower.index("base_label") if "base_label" in lower else -1

            if manual_idx == -1 and base_idx == -1:
                logging.warning(
                    "Manual labels CSV must contain 'manual_label' or 'base_label' column: %s", csv_path
                )
                return None

            for row in reader:
                if len(row) <= time_idx:
                    continue
                raw_t = row[time_idx]
                try:
                    t = float(raw_t)
                except (TypeError, ValueError):
                    continue

                manual_label = ""
                base_label = ""
                if manual_idx != -1 and len(row) > manual_idx:
                    manual_label = (row[manual_idx] or "").strip()
                if base_idx != -1 and len(row) > base_idx:
                    base_label = (row[base_idx] or "").strip()

                # Prefer manual_label if present; otherwise fall back to base_label.
                chosen = manual_label or base_label
                if not chosen:
                    continue

                # Normalize to the same coarse label space we use for predictions.
                # The interactive tool uses exactly 'S1', 'S2', or 'Noise', but we
                # keep this tolerant in case of minor variations.
                val = chosen.strip()
                if val.startswith("S1"):
                    norm = "S1"
                elif val.startswith("S2"):
                    norm = "S2"
                elif val.lower().startswith("noise"):
                    norm = "Noise"
                else:
                    # Ignore unknown labels rather than guessing.
                    continue

                key = f"{t:.3f}"
                labels_by_time[key] = norm

    except Exception as e:
        logging.error("Failed to read manual labels CSV '%s': %s", csv_path, e)
        return None

    if not labels_by_time:
        logging.info("Manual labels CSV found but contained no usable label rows: %s", csv_path)
        return None

    logging.info("Loaded %d manual peak labels from '%s' for validation.", len(labels_by_time), csv_name)
    return labels_by_time


def _build_predicted_labels_for_validation(
    analysis_data: Dict, sample_rate: int
) -> Dict[str, str]:
    """
    Builds a mapping of rounded time_sec ('%.3f') -> coarse label ('S1'/'S2'/'Noise'/'Unknown')
    from the analysis debug info.
    """
    debug_info = analysis_data.get("peak_classifications", {})
    labels_by_time: Dict[str, str] = {}

    if not debug_info:
        return labels_by_time

    for peak_idx, entry in debug_info.items():
        try:
            t = float(peak_idx) / float(sample_rate)
        except Exception:
            continue
        key = f"{t:.3f}"
        label = _simple_label_from_debug(entry)
        # In the rare case of duplicated times at this rounding, we let the
        # last one win; this mirrors how the JS importer behaves.
        labels_by_time[key] = label

    return labels_by_time


def _append_validation_results_row(
    regression_log_path: Optional[str],
    audio_file_path: str,
    manual_labels: Dict[str, str],
    predicted_labels: Dict[str, str],
) -> None:
    """
    Compares manual vs predicted labels and logs a per-file summary.

    For any file with discrepancies, it also logs each mismatched time
    with the algorithm's label and the correct manual label.
    """
    if not manual_labels:
        return

    all_truth_keys = set(manual_labels.keys())
    all_pred_keys = set(predicted_labels.keys())

    matched_keys = all_truth_keys & all_pred_keys
    missing_keys = all_truth_keys - all_pred_keys   # manual label exists, prediction missing
    extra_keys = all_pred_keys - all_truth_keys     # prediction exists, no manual label

    correct = 0
    mismatched = 0
    for k in matched_keys:
        if manual_labels[k] == predicted_labels.get(k):
            correct += 1
        else:
            mismatched += 1

    manual_count = len(all_truth_keys)
    predicted_count = len(all_pred_keys)
    missing = len(missing_keys)
    extra = len(extra_keys)
    total_errors = mismatched + missing + extra

    audio_name = os.path.basename(audio_file_path)

    # --- Console logging summary ---
    if total_errors == 0:
        logging.info(
            "Manual label validation for '%s': all %d peaks matched.",
            audio_name,
            manual_count,
        )
    else:
        logging.info(
            "Manual label validation for '%s': manual=%d, predicted=%d, matched=%d, "
            "correct=%d, mismatched=%d, missing=%d, extra=%d, total_errors=%d",
            audio_name,
            manual_count,
            predicted_count,
            len(matched_keys),
            correct,
            mismatched,
            missing,
            extra,
            total_errors,
        )

    # --- Optional regression-testing markdown log ---
    log_file = None
    if regression_log_path:
        try:
            log_file = open(regression_log_path, "a", encoding="utf-8")
            log_file.write(f"## {audio_name}\n\n")
            log_file.write(f"- **Manual peaks**: {manual_count}\n")
            log_file.write(f"- **Predicted peaks**: {predicted_count}\n")
            log_file.write(f"- **Matched peaks**: {len(matched_keys)}\n")
            log_file.write(f"- **Correct matches**: {correct}\n")
            log_file.write(f"- **Label mismatches**: {mismatched}\n")
            log_file.write(f"- **Missing detections**: {missing}\n")
            log_file.write(f"- **Extra detections**: {extra}\n")
            log_file.write(f"- **Total errors**: {total_errors}\n\n")

            if total_errors == 0:
                log_file.write("All peaks matched between algorithm and manual labels.\n\n")
        except Exception as e:
            logging.error(
                "Failed to append validation summary to regression log '%s': %s",
                regression_log_path,
                e,
            )
            log_file = None

    # Detailed per-peak differences: algorithm vs correct label.
    # 1) Label mismatches where both sides have a label.
    if mismatched > 0:
        logging.info("  Label mismatches for '%s':", audio_name)
        if log_file:
            log_file.write("### Label mismatches\n")
        for k in sorted(matched_keys):
            true_label = manual_labels[k]
            pred_label = predicted_labels.get(k, "Unknown")
            if true_label == pred_label:
                continue
            try:
                t = float(k)
            except ValueError:
                t = k
            formatted_t = f"{t:.3f}" if isinstance(t, (float, int)) else str(t)
            logging.info("    t=%s s  manual=%s  predicted=%s", formatted_t, true_label, pred_label)
            if log_file:
                log_file.write(
                    f"- t={formatted_t} s -- **manual**: {true_label}, **predicted**: {pred_label}\n"
                )
        if log_file:
            log_file.write("\n")

    # 2) Manual peaks that had no corresponding prediction.
    if missing > 0:
        logging.info("  Missing detections for '%s' (manual label but no predicted peak):", audio_name)
        if log_file:
            log_file.write("### Missing detections (manual label but no predicted peak)\n")
        for k in sorted(missing_keys):
            true_label = manual_labels[k]
            try:
                t = float(k)
            except ValueError:
                t = k
            formatted_t = f"{t:.3f}" if isinstance(t, (float, int)) else str(t)
            logging.info("    t=%s s  manual=%s  predicted=<none>", formatted_t, true_label)
            if log_file:
                log_file.write(
                    f"- t={formatted_t} s -- **manual**: {true_label}, **predicted**: <none>\n"
                )
        if log_file:
            log_file.write("\n")

    # 3) Extra predictions that have no manual label.
    if extra > 0:
        logging.info("  Extra detections for '%s' (predicted peak but no manual label):", audio_name)
        if log_file:
            log_file.write("### Extra detections (predicted peak but no manual label)\n")
        for k in sorted(extra_keys):
            pred_label = predicted_labels.get(k, "Unknown")
            try:
                t = float(k)
            except ValueError:
                t = k
            formatted_t = f"{t:.3f}" if isinstance(t, (float, int)) else str(t)
            logging.info("    t=%s s  manual=<none>  predicted=%s", formatted_t, pred_label)
            if log_file:
                log_file.write(
                    f"- t={formatted_t} s -- **manual**: <none>, **predicted**: {pred_label}\n"
                )
        if log_file:
            log_file.write("\n")

    if log_file:
        try:
            log_file.flush()
            log_file.close()
        except Exception:
            pass
