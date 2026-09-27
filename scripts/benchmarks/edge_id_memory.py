#!/usr/bin/env python3
"""What storing edge IDs outside LMDB costs in memory, and what it saves in time.

Compares four ways to hold the original ID of every edge:

* ``lmdb``: ``edge_ids.lmdb``, one key per edge (the format before
  ``EdgeIdStore``).
* ``mmap``: ``EdgeIdStore`` memory-mapped, as the server loads it.
* ``in_memory``: ``EdgeIdStore`` copied into RAM, as it is loaded with
  ``GANDALF_LOAD_MMAPS_INTO_MEMORY``.
* ``pylist``: a Python list of ``str`` (the old ``edge_ids.pkl``).

For each, in a fresh process: the size on disk, and the process's private
(``RssAnon``) and file-backed (``RssFile``) resident memory after loading,
after one query-sized batch lookup, and after reading every ID.  File-backed
pages are the page cache: every worker that maps the same file shares one
copy, and the kernel can evict them under pressure.  Private pages are paid
once per gunicorn worker.  Also the time of a batch lookup of ``--batch``
random edges, decoded to ``str`` (what ``get_edge_ids_batch`` returns) and,
for the blob, left as the JSON a response writes.

The IDs come from a built graph (``--graph``, either format) or are made up
(``--edges`` IDs in the ``--id-format`` shape).  Figures are per edge as
well, so a run on a sample extrapolates to the full graph.

Usage::

    python scripts/benchmarks/edge_id_memory.py --graph /data/graph_mmap
    python scripts/benchmarks/edge_id_memory.py --edges 5000000 --id-format uuid

Linux only (reads ``/proc/self/status``).
"""

import argparse
import hashlib
import json
import pickle
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import lmdb
import numpy as np

from gandalf.edge_id_store import BLOB_FILE, OFFSETS_FILE, EdgeIdStore

VARIANTS = ("lmdb", "mmap", "in_memory", "pylist")

#: Made-up ID shapes: a ``urn:uuid`` (45 characters), a SHA-1 hex digest
#: (40), and a short source-prefixed counter (about 20).
ID_FORMATS = {
    "uuid": lambda i: f"urn:uuid:{uuid.UUID(int=i * 2654435761 + 1)}",
    "sha1": lambda i: hashlib.sha1(i.to_bytes(8, "little")).hexdigest(),
    "short": lambda i: f"infores:x/{i}",
}

_KEY = struct.Struct(">I")


def rss() -> dict:
    """This process's private and file-backed resident memory, in bytes."""
    fields = {}
    with open("/proc/self/status") as f:
        for line in f:
            name, _, value = line.partition(":")
            if name in ("RssAnon", "RssFile"):
                fields[name] = int(value.split()[0]) * 1024
    return fields


def read_graph_ids(graph: Path) -> list:
    """Every edge's ID from a built graph, in forward-CSR order."""
    num_edges = len(np.load(graph / "fwd_targets.npy", mmap_mode="r"))
    if (graph / OFFSETS_FILE).exists():
        with open(graph / "metadata.pkl", "rb") as f:
            store = EdgeIdStore.load(graph, pickle.load(f))
        return [store.get(i) for i in range(num_edges)]
    ids: list = [None] * num_edges
    env = lmdb.open(str(graph / "edge_ids.lmdb"), readonly=True, lock=False)
    with env.begin(buffers=True) as txn:
        for key, value in txn.cursor():
            ids[_KEY.unpack(bytes(key))[0]] = bytes(value).decode("utf-8")
    env.close()
    return ids


def write_lmdb(path: Path, ids: list) -> None:
    """Write *ids* as the edge-ID LMDB was written: one key per edge."""
    path.mkdir(parents=True)
    size = sum(len(i) for i in ids if i) * 4 + (1 << 30)
    env = lmdb.open(str(path), map_size=size, max_dbs=0, readahead=False)
    with env.begin(write=True) as txn:
        for idx, eid in enumerate(ids):
            if eid:
                txn.put(_KEY.pack(idx), eid.encode("utf-8"), append=True)
    env.close()


def disk_size(path: Path) -> int:
    """Bytes a file or directory of files takes on disk (allocated blocks)."""
    files = [path] if path.is_file() else list(path.iterdir())
    return sum(f.stat().st_blocks * 512 for f in files)


def prepare(ids: list, work: Path) -> dict:
    """Write every variant's files under *work*; return their disk sizes."""
    write_lmdb(work / "edge_ids.lmdb", ids)
    store = EdgeIdStore.from_ids(ids)
    store.save(work)
    with open(work / "store_metadata.pkl", "wb") as f:
        pickle.dump(store.metadata(), f)
    with open(work / "edge_ids.pkl", "wb") as f:
        pickle.dump(ids, f, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "lmdb": disk_size(work / "edge_ids.lmdb"),
        "mmap": disk_size(work / BLOB_FILE) + disk_size(work / OFFSETS_FILE),
        "in_memory": disk_size(work / BLOB_FILE) + disk_size(work / OFFSETS_FILE),
        "pylist": disk_size(work / "edge_ids.pkl"),
    }


def measure(variant: str, work: Path, num_edges: int, batch: int, repeat: int):
    """Load one variant in this process and report its memory and timings."""
    rng = np.random.default_rng(0)
    query = np.unique(rng.integers(0, num_edges, size=batch))
    out: dict = {"before": rss()}

    if variant == "lmdb":
        env = lmdb.open(
            str(work / "edge_ids.lmdb"),
            readonly=True,
            max_dbs=0,
            map_size=256 << 30,
            readahead=False,
            lock=False,
        )

        def as_str(indices):
            results = {}
            with env.begin(buffers=True) as txn:
                for idx in sorted({int(i) for i in indices}):
                    val = txn.get(_KEY.pack(idx))
                    if val is not None:
                        results[idx] = bytes(val).decode("utf-8")
            return results

        as_json = None

        def touch_all():
            with env.begin(buffers=True) as txn:
                for _key, value in txn.cursor():
                    len(value)

    elif variant == "pylist":
        with open(work / "edge_ids.pkl", "rb") as f:
            ids = pickle.load(f)

        def as_str(indices):
            return {i: ids[i] for i in indices.tolist() if ids[i]}

        as_json = None

        def touch_all():
            pass  # already all in memory

    else:
        with open(work / "store_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        store = EdgeIdStore.load(work, metadata, in_memory=variant == "in_memory")
        as_str = store.get_batch
        as_json = store.get_json_batch

        def touch_all():
            # One read per page of the blob, and every offset.
            np.frombuffer(store._blob, dtype=np.uint8)[::4096].sum()
            store.offsets.sum()

    out["loaded"] = rss()

    touch_all()
    out["resident"] = rss()

    def best(fn):
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            result = fn(query)
            times.append(time.perf_counter() - t0)
            del result
        return min(times)

    out["batch_str_s"] = best(as_str)
    if as_json is not None:
        out["batch_json_s"] = best(as_json)
    out["batch_edges"] = len(query)
    return out


def report(sizes: dict, results: dict, num_edges: int, id_bytes: int) -> None:
    """Print the comparison, in MB and bytes per edge."""
    mb = 1 << 20

    def cell(value: int) -> str:
        return f"{value / mb:9.1f} MB {value / num_edges:6.1f} B/e"

    print(
        f"\n{num_edges:,} edges; IDs average {id_bytes / num_edges:.1f} bytes "
        f"of UTF-8\n"
    )
    print(f"{'':10} {'on disk':>20}")
    for v in VARIANTS:
        print(f"{v:10} {cell(sizes[v])}")

    for stage, label in (("loaded", "just loaded"), ("resident", "every ID read")):
        print(f"\nResident memory, {label} (minus before loading)")
        print(f"{'':10} {'private (per worker)':>24} {'file-backed (shared)':>24}")
        for v in VARIANTS:
            r, base = results[v][stage], results[v]["before"]
            anon = r["RssAnon"] - base["RssAnon"]
            file = r["RssFile"] - base["RssFile"]
            print(f"{v:10} {cell(anon):>24} {cell(file):>24}")

    batch = results["mmap"]["batch_edges"]
    print(f"\nBatch lookup of {batch:,} random edges (best of runs)")
    for v in VARIANTS:
        r = results[v]
        line = f"{v:10} str: {r['batch_str_s']:7.3f}s ({r['batch_str_s'] / batch * 1e9:5.0f} ns/e)"
        if "batch_json_s" in r:
            line += (
                f"   json: {r['batch_json_s']:7.3f}s "
                f"({r['batch_json_s'] / batch * 1e9:5.0f} ns/e)"
            )
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--graph", type=Path, help="read the IDs of a built graph")
    source.add_argument("--edges", type=int, default=2_000_000)
    parser.add_argument("--id-format", choices=sorted(ID_FORMATS), default="uuid")
    parser.add_argument(
        "--batch", type=int, default=2_400_000, help="edges per batch lookup"
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--workdir", type=Path, help="where to write (default: tmp)")
    parser.add_argument("--json", type=Path, help="also write the results here")
    parser.add_argument("--measure", choices=VARIANTS, help=argparse.SUPPRESS)
    parser.add_argument("--num-edges", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.measure:
        result = measure(
            args.measure, args.workdir, args.num_edges, args.batch, args.repeat
        )
        print(json.dumps(result))
        return

    if args.graph:
        ids = read_graph_ids(args.graph)
    else:
        make = ID_FORMATS[args.id_format]
        ids = [make(i) for i in range(args.edges)]
    num_edges = len(ids)
    id_bytes = sum(len(i.encode("utf-8")) for i in ids if i)

    work = args.workdir or Path(tempfile.mkdtemp(prefix="edge_id_memory_"))
    work.mkdir(parents=True, exist_ok=True)
    try:
        sizes = prepare(ids, work)
        del ids
        results = {}
        for variant in VARIANTS:
            proc = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--measure",
                    variant,
                    "--workdir",
                    str(work),
                    "--num-edges",
                    str(num_edges),
                    "--batch",
                    str(min(args.batch, num_edges)),
                    "--repeat",
                    str(args.repeat),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            results[variant] = json.loads(proc.stdout)
        report(sizes, results, num_edges, id_bytes)
        if args.json:
            args.json.write_text(
                json.dumps(
                    {
                        "num_edges": num_edges,
                        "id_bytes": id_bytes,
                        "disk": sizes,
                        "results": results,
                    },
                    indent=2,
                )
            )
    finally:
        if not args.workdir:
            shutil.rmtree(work)


if __name__ == "__main__":
    main()
