# Scripts Directory

This directory contains reproduction scripts and tools for debugging pgvectorscale issues.

## Issue #193 Reproduction

Files related to reproducing GitHub issue #193 (page corruption with concurrent diskann inserts):

- `issue_193_repro.py` - Python reproduction script
- `requirements.txt` - Python dependencies
- `run_repro.sh` - Helper script to run reproductions
- `venv/` - Python virtual environment

### Setup

1. Install Python dependencies:
   ```bash
   cd scripts
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Ensure pgvectorscale is installed in your PostgreSQL instance:
   ```bash
   psql -h localhost -p 28817 -d postgres -c "CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;"
   ```

### Usage

Run the reproduction script:
```bash
# Single-threaded (should work fine)
./run_repro.sh --batch-size 100 --batches 10 --parallelism 1

# Multi-threaded (reproduces the issue)
./run_repro.sh --batch-size 100 --batches 50 --parallelism 4
```

### Issue Details

The reproduction script demonstrates a page corruption assertion error when performing concurrent vector insertions with diskann indexes:

```
assertion failed: (*header).pd_special >= SizeOfPageHeaderData as u16
```

- **Error occurs**: With parallelism > 1
- **Error does not occur**: With parallelism = 1  
- **Affected versions**: pgvectorscale 0.5.1-0.6.0, TimescaleDB 2.17.2+
- **Vector dimensions**: Non-power-of-2 sizes (e.g., 1024, 513)