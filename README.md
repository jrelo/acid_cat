# acid_cat

WAV metadata explorer.
Scans RIFF chunks to find ACID info (BPM, beats, meter, root), SMPL loop points, BWF `bext`, `LIST/INFO` tags, cues, and more.  
Works great for hunting **acidized loops** and building datasets for ML experiments.

## Features

- Robust EOF scanning (won't miss trailing chunks)
- Summary mode: BPM, beats, expected duration, SMPL loop points
- `--all` mode: full per-chunk key/value dump to CSV
- `--kinds` mode: quick "what metadata does each file have?"
- `--survey` mode: count chunk types across directories
- `--has acid,smpl,...` filter to target exactly what you want
- `-v` verbose printing in `--all` mode (and `-q` to suppress)

## Quick start

```bash
# Summary view of acidized files (prints to console + writes CSV)
python acid_cat.py "D:\Audio\Loops" --has acid -n 20

# Chunk kinds per file (prints + writes <name>_kinds.csv)
python acid_cat.py "D:\Audio\Loops" --kinds -n 50

# Full chunk dump to CSV (quiet)
python acid_cat.py "D:\Audio\Loops" --all --has acid -n 200 -q

# Full chunk dump and also print parsed rows
python acid_cat.py "D:\Audio\Loops" --all --has acid -n 10 -v

# Survey a large tree to find where metadata lives
python acid_cat.py "D:\Audio" --survey -n 2000
