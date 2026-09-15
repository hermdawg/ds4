# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "gguf>=0.17"]
# ///
"""Measure early-router predictions using ds4's existing Metal tensor dumps.

This is an offline diagnostic, not a prefetch implementation or speed benchmark.
The model always runs its original experts. Only small router matrices are read
for analysis; expert weights are never copied into Python memory.
"""

import argparse
from collections import OrderedDict
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np


FIELDS = {
    "ffn_norm": ("bin", "<f4", 4096),
    "hc_ffn_pre": ("bin", "<f4", 4096),
    "hc_attn_pre": ("bin", "<f4", 4096),
    "ffn_moe_logits": ("bin", "<f4", 256),
    "ffn_moe_topk": ("i32", "<i4", 6),
}
LAYERS, EXPERTS, K = 43, 256, 6


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def pack_dumps(directory):
    """Require complete layers at every recorded decode position."""
    index = {}
    prefill_files = []
    for name, (suffix, dtype, width) in FIELDS.items():
        files = {}
        for path in directory.glob(f"trace_{name}-*_pos*.{suffix}"):
            match = re.fullmatch(rf"trace_{name}-(\d+)_pos(\d+)\.{suffix}", path.name)
            if match:
                layer, pos = map(int, match.groups())
                row_bytes = np.dtype(dtype).itemsize * width
                size = path.stat().st_size
                # The same debug hook also emits whole prefill chunks. Those
                # files have multiple rows; never treat them as decode vectors.
                if size > row_bytes and size % row_bytes == 0:
                    prefill_files.append(path)
                    continue
                if size != row_bytes:
                    raise ValueError(f"Unexpected tensor size: {path}")
                files[pos, layer] = path
        if not files:
            raise ValueError(f"Missing dumps for {name}")
        index[name] = files
    keys = set(index["ffn_norm"])
    if any(set(files) != keys for files in index.values()):
        raise ValueError("Tensor dumps disagree on positions/layers")
    positions = sorted({pos for pos, _ in keys})
    if keys != {(p, l) for p in positions for l in range(LAYERS)}:
        raise ValueError("Incomplete layer coverage")
    if len(positions) < 2 or np.any(np.diff(positions) != 1):
        raise ValueError("Expected consecutive decode positions")
    arrays = {"positions": np.array(positions, dtype=np.int32)}
    for name, (_, dtype, width) in FIELDS.items():
        arrays[name] = np.stack([
            np.stack([np.fromfile(index[name][p, l], dtype=dtype)
                      for l in range(LAYERS)]) for p in positions
        ])
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"Nonfinite values in {name}")
    selected = arrays["ffn_moe_topk"]
    if np.any((selected < 0) | (selected >= EXPERTS)):
        raise ValueError("Invalid expert IDs")
    if np.any(np.diff(np.sort(selected, axis=-1), axis=-1) == 0):
        raise ValueError("Duplicate selected expert IDs")
    output = directory / "trace.npz"
    np.savez_compressed(output, **arrays)
    # Only remove our raw files after re-opening and checking the packed copy.
    with np.load(output) as packed:
        for name, values in arrays.items():
            if not np.array_equal(packed[name], values):
                raise ValueError(f"Packed trace mismatch: {name}")
    for files in index.values():
        for path in files.values():
            path.unlink()
    for path in prefill_files:
        path.unlink()
    return len(positions)


def collect(args):
    cases = json.loads(args.prompts.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "binary": str(args.binary.resolve()), "model": str(args.model.resolve()),
        "tokens_requested": args.tokens, "cache_target": "8GB",
        "context": 8192, "prefill_chunk": 1024, "cases": [],
        "warning": "Diagnostic synchronization makes these timings unsuitable for benchmarking.",
    }
    manifest["binary_source_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.binary.resolve().parent, text=True).strip()
    for case in cases:
        directory = args.output / case["id"]
        directory.mkdir(exist_ok=False)
        env = dict(os.environ)
        # Do not inherit unrelated inference diagnostic settings.
        for key in list(env):
            if key.startswith("DS4_"):
                del env[key]
        env.update(DS4_METAL_GRAPH_DUMP_PREFIX=str(directory.resolve() / "trace"),
                   DS4_METAL_GRAPH_DUMP_NAME=",".join(FIELDS))
        command = [str(args.binary.resolve()), "-m", str(args.model.resolve()),
                   "--ssd-streaming", "--ssd-streaming-cache-experts", "8GB",
                   "--ctx", "8192", "--prefill-chunk", "1024", "--nothink",
                   "--temp", "0", "-n", str(args.tokens), "-p", case["prompt"]]
        start = time.monotonic()
        with (directory / "stdout.txt").open("w") as out, (directory / "stderr.txt").open("w") as err:
            subprocess.run(command, env=env, stdout=out, stderr=err, check=True, timeout=600)
        count = pack_dumps(directory)
        result = {**case, "positions": count, "elapsed_seconds": round(time.monotonic() - start, 3)}
        manifest["cases"].append(result)
        write_json(args.output / "manifest.json", manifest)
        print(json.dumps(result), flush=True)


def wilson_lower(hits, count):
    """Descriptive 95% binomial lower bound; routing events are not independent."""
    if count == 0:
        return None
    z = 1.959963984540054
    p = hits / count
    return (p + z*z/(2*count) - z*math.sqrt(p*(1-p)/count + z*z/(4*count*count))) / (1+z*z/count)


def summarize(confidence, correct, threshold):
    keep = confidence >= threshold
    count = int(keep.sum())
    hits = int(correct[keep].sum())
    return {
        "opportunities": len(correct), "issued": count, "correct": hits,
        "precision": hits/count if count else None,
        "issue_fraction": count/len(correct) if len(correct) else 0,
        "binomial_wilson_lower_95": wilson_lower(hits, count),
    }


def calibrate(confidence, correct, target, minimum):
    """Select on calibration prompts only; issue all candidates tied at cutoff."""
    if len(confidence) == 0:
        return None
    order = np.argsort(-confidence, kind="stable")
    c, y = confidence[order], correct[order]
    totals = np.arange(1, len(y)+1)
    precision = np.cumsum(y) / totals
    ends = np.r_[c[:-1] != c[1:], True]
    acceptable = np.flatnonzero((precision >= target) & (totals >= minimum) & ends)
    return float(c[acceptable[-1]]) if len(acceptable) else None


def summarize_prompts(events, threshold):
    results = {}
    for name in sorted({event[0] for event in events}):
        rows = [event for event in events if event[0] == name]
        results[name] = summarize(np.concatenate([r[1] for r in rows]),
                                  np.concatenate([r[2] for r in rows]), threshold)
    return results


def cache_presence(selected, capacity):
    """Snapshot a shadow global LRU before each layer, not ds4's actual cache.

    Initial prefill cache contents are unavailable. Analysis skips the first 16
    decode positions. Expert keys include layer: IDs alone do not share weights.
    """
    present = np.zeros((len(selected), LAYERS, LAYERS, EXPERTS), dtype=bool)
    cache = OrderedDict()
    for t, token in enumerate(selected):
        for layer, experts in enumerate(token):
            for cached_layer, expert in cache:
                present[t, layer, cached_layer, expert] = True
            for expert in experts:
                key = layer, int(expert)
                cache.pop(key, None)
                cache[key] = None
                while len(cache) > capacity:
                    cache.popitem(last=False)
    return present


def analyze(args):
    from gguf import GGUFReader

    manifest = json.loads((args.input / "manifest.json").read_text())
    cases = manifest["cases"]
    if {c["split"] for c in cases} != {"calibration", "test"}:
        raise ValueError("Need distinct calibration and test prompts")
    reader = GGUFReader(str(args.model))
    tensors = {t.name: t for t in reader.tensors}
    # Access just the router and normalization tensors in the mmap-backed GGUF.
    weights, norms, biases = [], [], []
    for l in range(LAYERS):
        weights.append(np.asarray(tensors[f"blk.{l}.ffn_gate_inp.weight"].data, dtype=np.float32))
        norms.append(np.asarray(tensors[f"blk.{l}.ffn_norm.weight"].data, dtype=np.float32))
        bias = tensors.get(f"blk.{l}.exp_probs_b.bias")
        biases.append(np.asarray(bias.data, dtype=np.float32) if bias is not None else np.zeros(EXPERTS, dtype=np.float32))
    if any(w.shape != (EXPERTS, 4096) for w in weights):
        raise ValueError("This probe only supports the V4 Flash 43/256/6 layout")
    eps_field = reader.fields["deepseek4.attention.layer_norm_rms_epsilon"]
    epsilon = float(eps_field.parts[eps_field.data[0]].item())
    events = {}
    validation = []
    recalls = {}
    methods = ["pre_attention", "ahead1", "ahead2", "ahead1_renorm", "ahead2_renorm"]
    for case in cases:
        with np.load(args.input / case["id"] / "trace.npz") as trace:
            x = trace["ffn_norm"]
            raw = trace["hc_ffn_pre"]
            preattn = trace["hc_attn_pre"]
            selected = trace["ffn_moe_topk"]
            gpu_logits = trace["ffn_moe_logits"]
        present = cache_presence(selected, args.cache_slots)
        n = len(x)
        for target in range(3, LAYERS):  # Hash layers are known from token IDs.
            w, norm, bias = weights[target], norms[target], biases[target]
            actual = selected[:, target]
            logits = x[:, target] @ w.T
            scores = np.sqrt(np.logaddexp(0, logits)) + bias
            recomputed = np.argsort(-scores, axis=-1, kind="stable")[:, :K]
            exact = np.all(np.sort(recomputed, axis=-1) == np.sort(actual, axis=-1), axis=-1)
            validation.append({"case": case["id"], "layer": target,
                               "positions": n, "exact_sets": int(exact.sum()),
                               "max_logit_error": float(np.max(np.abs(logits-gpu_logits[:, target])))})
            # Reject bad reconstruction before interpreting any early prediction.
            if exact.mean() < .99:
                raise ValueError(f"Router reconstruction failed: {case['id']} layer {target}: {exact.mean()}")
            for method in methods:
                ahead = 0 if method == "pre_attention" else int(method[5])
                source = target - ahead
                if source < 0:
                    continue
                if method == "pre_attention" or method.endswith("renorm"):
                    value = preattn[:, target] if ahead == 0 else raw[:, source]
                    value = value / np.sqrt(np.mean(value*value, axis=-1, keepdims=True) + epsilon) * norm
                else:
                    value = x[:, source]
                predicted_scores = np.sqrt(np.logaddexp(0, value @ w.T)) + bias
                ranking = np.argsort(-predicted_scores, axis=-1, kind="stable")
                correct6 = np.any(ranking[:, :K, None] == actual[:, None, :], axis=-1)
                start = min(16, n)
                recalls.setdefault((method, case["split"]), []).extend(correct6[start:].mean(axis=-1).tolist())
                boundary = np.take_along_axis(predicted_scores, ranking[:, K:K+1], axis=-1)[:, 0]
                scale = predicted_scores.std(axis=-1) + 1e-9
                for scope in ("all", "shadow_lru_miss"):
                    eligible = np.ones((n, EXPERTS), dtype=bool) if scope == "all" else ~present[:, source, target]
                    available = np.take_along_axis(eligible, ranking[:, :K], axis=-1)
                    valid = available.any(axis=-1)
                    slot = available.argmax(axis=-1)
                    candidate = ranking[np.arange(n), slot]
                    confidence = (predicted_scores[np.arange(n), candidate] - boundary) / scale
                    correct = np.any(candidate[:, None] == actual, axis=-1)
                    valid[:start] = False
                    key = method, scope, case["split"]
                    events.setdefault(key, []).append((case["id"], confidence[valid], correct[valid]))
        print(f"Analyzed {case['id']}: {n} positions", flush=True)
    output = {"manifest": manifest, "settings": {"target_precision": args.precision,
              "minimum_calibration_predictions": args.minimum, "shadow_lru_slots": args.cache_slots,
              "warmup_positions_discarded": 16}, "validation": {
                  "sets": sum(v["positions"] for v in validation),
                  "exact_sets": sum(v["exact_sets"] for v in validation),
                  "max_logit_error": max(v["max_logit_error"] for v in validation)},
              "results": [], "limitations": [
                  "Prediction quality only; no prefetch I/O or speedup measured.",
                  "LRU cache is a shadow simulation, not ds4's production policy or macOS page cache.",
                  "A few short prompts cannot establish near-perfect precision on general workloads.",
                  "Binomial bounds ignore correlation between positions and layers; they are descriptive only.",
                  "Confidence is a score margin, not a calibrated probability."]}
    for method in methods:
        for scope in ("all", "shadow_lru_miss"):
            splits = {}
            for split in ("calibration", "test"):
                data = events[method, scope, split]
                splits[split] = np.concatenate([d[1] for d in data]), np.concatenate([d[2] for d in data])
            threshold = calibrate(*splits["calibration"], args.precision, args.minimum)
            result = {"method": method, "scope": scope, "confidence_threshold": threshold,
                      "top6_recall_test": float(np.mean(recalls[method, "test"])),
                      "unfiltered_test": summarize(*splits["test"], -math.inf)}
            if threshold is not None:
                result["calibration"] = summarize(*splits["calibration"], threshold)
                result["test"] = summarize(*splits["test"], threshold)
                result["test_by_prompt"] = summarize_prompts(events[method, scope, "test"], threshold)
            output["results"].append(result)
    write_json(args.output, output)
    print(json.dumps({"validation": output["validation"], "results": output["results"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("collect")
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=128)
    p.set_defaults(run=collect)
    p = commands.add_parser("analyze")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--precision", type=float, default=.99)
    p.add_argument("--minimum", type=int, default=200)
    p.add_argument("--cache-slots", type=int, default=701)
    p.set_defaults(run=analyze)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
