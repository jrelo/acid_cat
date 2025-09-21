# acid_cat

WAV metadata explorer.  
Scans RIFF chunks to find ACID info (BPM, beats, meter, root), SMPL loop points, BWF `bext`, `LIST/INFO` tags, cues, and more.  
Writes out to CSV.
Useful for hunting **acidized loops** and building datasets for ML experiments.

## Features

- Robust EOF scanning (won’t miss trailing chunks)
- Summary mode: BPM, beats, expected duration, SMPL loop points
- `--all` mode: full per-chunk key/value dump to CSV
- `--kinds` mode: quick overview of what metadata each file has
- `--survey` mode: count chunk types across a directory tree
- `--has acid,smpl,...` filter to target exactly what you want
- `--fallback` mode: estimate BPM/key using librosa if no ACID/SMPL metadata is found
- `-v` verbose printing in `--all` mode (and `-q` to suppress)

## Installation

Clone the repo and install dependencies:

    git clone https://github.com/jrelo/acid_cat.git
    cd acid_cat
    pip install -r requirements.txt

Requirements are minimal:

    numpy
    pandas
    librosa

(plus standard library modules)

## Usage examples

    # Summary view of acidized files (prints to console + writes CSV)
    python acid_cat.py "D:\Audio\Loops" --has acid -n 20

    # Chunk kinds per file (prints + writes <name>_kinds.csv)
    python acid_cat.py "D:\Audio\Loops" --kinds -n 50

    # Full chunk dump to CSV (quiet)
    python acid_cat.py "D:\Audio\Loops" --all --has acid -n 200 -q

    # Full chunk dump and also print parsed rows
    python acid_cat.py "D:\Audio\Loops" --all --has acid -n 10 -v

    # Survey a large tree to see what metadata appears where
    python acid_cat.py "D:\Audio" --survey -n 2000

    # Estimate BPM/key using librosa when ACID/SMPL metadata is missing
    python acid_cat.py "D:\Audio\Loops" --fallback -n 100
