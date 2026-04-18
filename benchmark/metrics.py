import numpy as np
from scipy.stats import pearsonr
from dataset.schemas import DIMENSIONS, WEIGHTS


def compute_metrics(results: list[dict]) -> dict:
    """
    results: list of {"id", "content", "haiku": scores_dict, "finetuned": scores_dict, "base": scores_dict}
    Ground truth = haiku scores.
    Returns nested dict of per-dimension and aggregate metrics for finetuned and base.
    """
    output = {}

    for model_key in ["finetuned", "base"]:
        model_metrics = {}
        for dim in DIMENSIONS:
            haiku_vals = np.array([r["haiku"][dim] for r in results])
            model_vals = np.array([r[model_key][dim] for r in results])

            mae = float(np.mean(np.abs(haiku_vals - model_vals)))
            exact = float(np.mean(haiku_vals == model_vals))
            within_1 = float(np.mean(np.abs(haiku_vals - model_vals) <= 1))
            corr, _ = pearsonr(haiku_vals, model_vals)

            model_metrics[dim] = {
                "mae": round(mae, 3),
                "exact_agreement": round(exact, 3),
                "within_1_agreement": round(within_1, 3),
                "pearson_r": round(float(corr), 3),
            }

        # Composite MAE
        haiku_composites = np.array([r["haiku"].get("composite_score", 0.0) for r in results])
        model_composites = np.array([r[model_key].get("composite_score", 0.0) for r in results])
        composite_mae = float(np.mean(np.abs(haiku_composites - model_composites)))

        # Tier accuracy
        def tier(c): return "ready" if c >= 9.25 else ("below_target" if c >= 8.0 else "failed_floor")
        tier_match = sum(tier(h) == tier(m) for h, m in zip(haiku_composites, model_composites))

        # never_list F1
        haiku_nlv = np.array([r["haiku"].get("never_list_violation", False) for r in results])
        model_nlv = np.array([r[model_key].get("never_list_violation", False) for r in results])
        tp = int(np.sum(haiku_nlv & model_nlv))
        fp = int(np.sum(~haiku_nlv & model_nlv))
        fn = int(np.sum(haiku_nlv & ~model_nlv))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        model_metrics["aggregate"] = {
            "mean_mae": round(float(np.mean([model_metrics[d]["mae"] for d in DIMENSIONS])), 3),
            "mean_within_1": round(float(np.mean([model_metrics[d]["within_1_agreement"] for d in DIMENSIONS])), 3),
            "composite_mae": round(composite_mae, 3),
            "tier_accuracy": round(tier_match / len(results), 3),
            "never_list_f1": round(f1, 3),
            "mean_latency_ms": round(float(np.mean([r[model_key].get("latency_ms", 0) for r in results])), 1),
        }
        output[model_key] = model_metrics

    return output
