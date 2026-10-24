#!/usr/bin/env python3
"""Fail closed on common secret/infrastructure material in tracked or staged blobs."""
import argparse
from pathlib import PurePosixPath
import re
import subprocess

def git(*args):
    return subprocess.check_output(["git", *args])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    findings = []
    rules = [
        ("private key", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
        ("service token", re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{20,})")),
        ("credential assignment", re.compile(rb"(?i)(?:password|passwd|api_key|access_token)\\s*[:=]\\s*[\\\"'][^\\\"'\\s]{6,}[\\\"']")),
        ("private absolute path", re.compile(rb"/(?:home|Users|mnt/data)/[A-Za-z0-9_.-]+")),
        ("IPv4 address", re.compile(rb"(?<![0-9])(?:[0-9]{1,3}\\.){3}[0-9]{1,3}(?![0-9])")),
        ("SSH connection", re.compile(rb"s" + rb"sh://|[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+:")),
    ]
    forbidden = {"runs", "runtime", ".tmp", ".orchestrator", "data", "datasets",
                 "models", "weights", ".dsh", ".codex", ".venv"}
    suffixes = {".pdf", ".jsonl", ".log", ".pem", ".key", ".safetensors", ".pt", ".pth", ".bin", ".bundle"}
    revisions = git("rev-list", "--all").decode().split() if args.history else [None]
    seen = set()
    for revision in revisions:
        paths = (git("ls-tree", "-r", "--name-only", revision) if revision
                 else git("ls-files")).decode().splitlines()
        for name in paths:
            spec = revision + ":" + name if revision else ":" + name
            oid = git("rev-parse", spec).strip()
            if (name, oid) in seen:
                continue
            seen.add((name, oid))
            path = PurePosixPath(name)
            if any(part in forbidden for part in path.parts) or path.suffix in suffixes:
                findings.append((name, "forbidden artifact path"))
            blob = git("show", spec)
            if len(blob) > 2 * 1024 * 1024:
                findings.append((name, "oversized blob"))
            for label, pattern in rules:
                if pattern.search(blob):
                    findings.append((name, label))
    for name, label in findings:
        print(f"{name}: {label}")  # never print matching values
    print(f"Audited {len(seen)} file versions; findings={len(findings)}")
    raise SystemExit(bool(findings))

if __name__ == "__main__":
    main()
