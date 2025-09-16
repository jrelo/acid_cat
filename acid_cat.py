#!/usr/bin/env python3
"""
acid_cat.py

Full WAV metadata explorer with robust EOF scanning.

Modes:
- Default (summary): ACID/SMPL summary + expected duration + "Other chunks" list
- --all: Enumerate all chunks, outputting one row per chunk/key/value
- --survey: Count chunk IDs across scanned files and output a summary CSV
- --kinds: List chunk IDs per file (quick "what metadata is here?" mode)
- --has: Filter to files that include any of the given chunk IDs (comma-separated)
- -v/--verbose: In --all mode, also print parsed rows to console (ignored if -q)

Examples:
  python acid_cat.py "D:\\Audio\\Loops" -n 200
  python acid_cat.py "D:\\Audio\\Loops" --all -n 50
  python acid_cat.py "D:\\Audio\\Loops" --survey -n 1000
  python acid_cat.py "D:\\Audio\\Loops" --kinds -n 200
  python acid_cat.py "D:\\Audio\\Loops" --kinds --has acid
  python acid_cat.py "D:\\Audio\\Loops" --all --has acid,smpl -v
"""

import argparse
import csv
import os
import re
import struct
import wave
import binascii
from collections import Counter

# --------------------------
# Helper: MIDI note -> name
# --------------------------
def midi_note_to_name(note_number):
    if note_number is None:
        return None
    NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (note_number // 12) - 1
    note = NOTES[note_number % 12]
    return f"{note}{octave}"

def safe_basename_for_csv(path_basename):
    # sluggy but readable: spaces -> _, strip weird chars
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", path_basename.strip())
    if not name.lower().endswith(".csv"):
        name += ".csv"
    return name

# --------------------------
# Core RIFF parser
# --------------------------
def parse_riff(filepath, enumerate_all=False):
    """
    Walk through WAV file and parse chunks.
    Returns:
      - results: list of (chunk_id, key, value) entries (used in --all mode)
      - meta:    summary dict with ACID/SMPL fields
      - seen:    ordered list of unique chunk IDs encountered
    """
    results = []
    meta = {
        "bpm": None,
        "acid_beats": None,
        "acid_root_note": None,
        "smpl_root_key": None,
        "smpl_loop_start": None,
        "smpl_loop_end": None,
    }
    seen_order = []
    seen_set = set()

    file_size = os.path.getsize(filepath)

    with open(filepath, "rb") as f:
        riff_header = f.read(12)
        if len(riff_header) < 12 or riff_header[0:4] != b'RIFF':
            return results, meta, seen_order  # Not a WAV

        pos = 12
        while pos < file_size:
            f.seek(pos)
            header = f.read(8)
            if len(header) < 8:
                break
            chunk_id = header[0:4]
            try:
                chunk_size = struct.unpack("<I", header[4:8])[0]
            except struct.error:
                break

            cid_str = chunk_id.decode("ascii", errors="ignore")
            if cid_str and cid_str not in seen_set:
                seen_set.add(cid_str)
                seen_order.append(cid_str)

            chunk_data = f.read(chunk_size)

            # --- Known chunk parsing ---
            if chunk_id == b'acid':
                # ACID chunk: version, root_note, (reserved), beats, meter_den, meter_num, tempo(float)
                try:
                    version, root_note, _, beats, meter_den, meter_num, tempo = struct.unpack("<IHHIII f", chunk_data)
                    meta["acid_root_note"] = root_note
                    meta["acid_beats"] = beats
                    meta["bpm"] = round(tempo, 2)
                    if enumerate_all:
                        results.append(("acid", "bpm", meta["bpm"]))
                        results.append(("acid", "beats", beats))
                        results.append(("acid", "root_note", midi_note_to_name(root_note)))
                        results.append(("acid", "meter", f"{meter_num}/{meter_den}"))
                        results.append(("acid", "version", version))
                except Exception as e:
                    if enumerate_all:
                        results.append(("acid", "error", str(e)))

            elif chunk_id == b'smpl':
                # SMPL chunk: sampler info + loop points
                try:
                    (
                        manufacturer, product, sample_period, midi_unity_note,
                        midi_pitch_fraction, smpte_format, smpte_offset,
                        sample_loops, sampler_data
                    ) = struct.unpack("<IIIIIIiiI", chunk_data[:36])
                    meta["smpl_root_key"] = midi_unity_note
                    if sample_loops > 0 and len(chunk_data) >= 36 + 24:
                        _, _, start, end, _, _ = struct.unpack("<IIIIII", chunk_data[36:60])
                        meta["smpl_loop_start"] = start
                        meta["smpl_loop_end"] = end
                    if enumerate_all:
                        results.append(("smpl", "root_key", midi_note_to_name(midi_unity_note)))
                        results.append(("smpl", "loops", sample_loops))
                        if meta["smpl_loop_start"] is not None:
                            results.append(("smpl", "loop_start", meta["smpl_loop_start"]))
                            results.append(("smpl", "loop_end", meta["smpl_loop_end"]))
                except Exception as e:
                    if enumerate_all:
                        results.append(("smpl", "error", str(e)))

            elif chunk_id == b'inst' and enumerate_all:
                # INST chunk: base note, detune, gain, key range, velocity range
                try:
                    if len(chunk_data) >= 7:
                        base = chunk_data[0]
                        detune = struct.unpack("<b", chunk_data[1:2])[0]
                        gain = struct.unpack("<b", chunk_data[2:3])[0]
                        low_note, high_note, low_vel, high_vel = chunk_data[3], chunk_data[4], chunk_data[5], chunk_data[6]
                        results.append(("inst", "base_note", midi_note_to_name(base)))
                        results.append(("inst", "detune_cents", detune))
                        results.append(("inst", "gain_db", gain))
                        results.append(("inst", "key_range", f"{midi_note_to_name(low_note)}-{midi_note_to_name(high_note)}"))
                        results.append(("inst", "vel_range", f"{low_vel}-{high_vel}"))
                    else:
                        results.append(("inst", "raw", binascii.hexlify(chunk_data).decode()))
                except Exception as e:
                    results.append(("inst", "error", str(e)))

            elif chunk_id == b'fmt ' and enumerate_all:
                # FMT chunk: codec, channels, rate, bits
                try:
                    if len(chunk_data) >= 16:
                        wFormatTag, nChannels, nSamplesPerSec, nAvgBytesPerSec, nBlockAlign, wBitsPerSample = struct.unpack("<HHIIHH", chunk_data[:16])
                        results.append(("fmt ", "format_tag", wFormatTag))
                        results.append(("fmt ", "channels", nChannels))
                        results.append(("fmt ", "sample_rate", nSamplesPerSec))
                        results.append(("fmt ", "bits_per_sample", wBitsPerSample))
                        results.append(("fmt ", "block_align", nBlockAlign))
                    else:
                        results.append(("fmt ", "raw", binascii.hexlify(chunk_data).decode()))
                except Exception as e:
                    results.append(("fmt ", "error", str(e)))

            elif chunk_id == b'fact' and enumerate_all:
                # FACT chunk: for non-PCM, first 4 bytes often sample length
                try:
                    if len(chunk_data) >= 4:
                        sample_length = struct.unpack("<I", chunk_data[:4])[0]
                        results.append(("fact", "sample_length", sample_length))
                    else:
                        results.append(("fact", "raw", binascii.hexlify(chunk_data).decode()))
                except Exception as e:
                    results.append(("fact", "error", str(e)))

            elif chunk_id == b'cue ' and enumerate_all:
                # CUE markers
                try:
                    num_cues = struct.unpack("<I", chunk_data[:4])[0]
                    for i in range(num_cues):
                        base = 4 + i * 24
                        cue_data = chunk_data[base: base + 24]
                        if len(cue_data) == 24:
                            _, _, _, _, _, sample_offset = struct.unpack("<IIIIII", cue_data)
                            results.append(("cue ", f"marker_{i}", sample_offset))
                except Exception as e:
                    results.append(("cue ", "error", str(e)))

            elif chunk_id == b'LIST' and enumerate_all:
                # LIST -> INFO tags (INAM, IART, ICMT, etc.)
                try:
                    if len(chunk_data) >= 4:
                        list_type = chunk_data[:4].decode("ascii", errors="ignore")
                        results.append(("LIST", "type", list_type))
                        pos_in_list = 4
                        while pos_in_list + 8 <= len(chunk_data):
                            sub_id = chunk_data[pos_in_list:pos_in_list+4].decode("ascii", errors="ignore")
                            sub_size = struct.unpack("<I", chunk_data[pos_in_list+4:pos_in_list+8])[0]
                            start = pos_in_list + 8
                            end = start + sub_size
                            if end > len(chunk_data):
                                break
                            sub_val = chunk_data[start:end].decode("ascii", errors="ignore").rstrip("\x00").strip()
                            results.append(("LIST", sub_id, sub_val))
                            pos_in_list = end
                            if sub_size % 2 == 1:
                                pos_in_list += 1
                    else:
                        results.append(("LIST", "raw", binascii.hexlify(chunk_data[:32]).decode()))
                except Exception:
                    results.append(("LIST", "raw", binascii.hexlify(chunk_data[:32]).decode()))

            elif chunk_id == b'bext' and enumerate_all:
                # Broadcast extension (BWF)
                try:
                    desc = chunk_data[0:256].decode("ascii", errors="ignore").rstrip("\x00").strip()
                    origin = chunk_data[256:288].decode("ascii", errors="ignore").rstrip("\x00").strip()
                    date = chunk_data[320:330].decode("ascii", errors="ignore").strip()
                    time = chunk_data[330:338].decode("ascii", errors="ignore").strip()
                    results.append(("bext", "description", desc))
                    results.append(("bext", "originator", origin))
                    results.append(("bext", "datetime", f"{date} {time}".strip()))
                except Exception:
                    results.append(("bext", "raw", binascii.hexlify(chunk_data[:32]).decode()))

            else:
                # Unknown or unparsed chunk
                if enumerate_all:
                    hex_preview = binascii.hexlify(chunk_data[:16]).decode()
                    results.append((cid_str, "raw", hex_preview))

            # Word alignment (chunks are word-aligned)
            pos += 8 + chunk_size
            if chunk_size % 2 == 1:
                pos += 1

    return results, meta, seen_order

def get_duration(filepath):
    try:
        with wave.open(filepath, 'rb') as wf:
            return round(wf.getnframes() / float(wf.getframerate()), 4)
    except:
        return None

# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="WAV Metadata Explorer (ACID/SMPL and more).")
    parser.add_argument("directory", help="Directory containing WAV files.")
    parser.add_argument("-o", "--output", help="Output CSV filename. Default: <dirname>_metadata.csv")
    parser.add_argument("-n", "--num", type=int, default=500, help="Number of WAV files to scan.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress console output.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose console output. In --all mode, echo chunk rows to the console (ignored if -q).")
    parser.add_argument("--all", action="store_true", help="Enumerate all chunks (not just ACID/SMPL).")
    parser.add_argument("--survey", action="store_true", help="Count chunk IDs across scanned files.")
    parser.add_argument("--kinds", "-k", action="store_true", help="Print just chunk kinds per file and write <name>_kinds.csv.")
    parser.add_argument("--has", help="Only include files that contain any of these chunk IDs (comma-separated, e.g. 'acid,smpl,LIST').")
    args = parser.parse_args()

    # Normalize output name, slugified a bit for Windows comfort
    default_base = os.path.basename(os.path.normpath(args.directory))
    output_csv = safe_basename_for_csv(args.output or (default_base + "_metadata.csv"))

    # Parse --has
    wanted = None
    if args.has:
        wanted = set([w.strip().upper() for w in args.has.split(",") if w.strip()])

    # SURVEY MODE
    if args.survey:
        counts = Counter()
        files_scanned = 0

        for root, _, files in os.walk(args.directory):
            for file in files:
                if file.lower().endswith(".wav"):
                    filepath = os.path.join(root, file)
                    _, _, seen = parse_riff(filepath, enumerate_all=False)
                    if wanted:
                        upper_seen = {s.upper() for s in seen}
                        if not (upper_seen & wanted):
                            continue
                    counts.update(seen)
                    files_scanned += 1
                    if files_scanned >= args.num:
                        break
            if files_scanned >= args.num:
                break

        print("\n== Chunk ID Survey ==")
        for cid, c in counts.most_common():
            print(f"{cid:6s} : {c} files")
        print(f"\n[INFO] Scanned {files_scanned} file(s). Found {len(counts)} unique chunk ID(s).")

        survey_csv = safe_basename_for_csv((args.output or (default_base + "_survey.csv")))
        with open(survey_csv, "w", newline="") as csvfile:
            w = csv.writer(csvfile)
            w.writerow(["chunk_id", "files_with_chunk"])
            for cid, c in counts.most_common():
                w.writerow([cid, c])
        print(f"[INFO] Wrote survey to {survey_csv}")
        return

    # KINDS MODE (quick list of metadata kinds per file)
    if args.kinds:
        kinds_rows = []
        count = 0
        for root, _, files in os.walk(args.directory):
            for file in files:
                if file.lower().endswith(".wav"):
                    filepath = os.path.join(root, file)
                    _, _, seen = parse_riff(filepath, enumerate_all=False)
                    if wanted:
                        upper_seen = {s.upper() for s in seen}
                        if not (upper_seen & wanted):
                            continue
                    kinds_rows.append({"filename": filepath, "chunks": ",".join(seen)})
                    if not args.quiet:
                        print(f"{os.path.basename(filepath)} : {', '.join(seen) if seen else '(none)'}")
                    count += 1
                    if count >= args.num:
                        break
            if count >= args.num:
                break

        kinds_csv = safe_basename_for_csv((args.output or (default_base + "_kinds.csv")))
        with open(kinds_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "chunks"])
            writer.writeheader()
            writer.writerows(kinds_rows)
        print(f"\n[INFO] Wrote kinds for {len(kinds_rows)} files to {kinds_csv}")
        return

    # REGULAR / --ALL MODES
    rows = []
    count = 0

    for root, _, files in os.walk(args.directory):
        for file in files:
            if file.lower().endswith(".wav"):
                filepath = os.path.join(root, file)
                all_chunks, meta, seen = parse_riff(filepath, enumerate_all=args.all)

                # Filter by --has if requested
                if wanted:
                    upper_seen = {s.upper() for s in seen}
                    if not (upper_seen & wanted):
                        continue

                duration = get_duration(filepath)

                if args.all:
                    if all_chunks:
                        for cid, key, val in all_chunks:
                            rows.append({"filename": filepath, "chunk": cid, "key": key, "value": val})

                        if args.verbose and not args.quiet:
                            print(f"\n=== {os.path.basename(filepath)} ===")
                            for cid, key, val in all_chunks:
                                print(f"  {cid}.{key} = {val}")
                    else:
                        rows.append({"filename": filepath, "chunk": "NONE", "key": "", "value": ""})
                        if args.verbose and not args.quiet:
                            print(f"\n=== {os.path.basename(filepath)} ===")
                            print("  (no parseable chunks)")
                else:
                    expected = diff = None
                    if meta["bpm"] and meta["acid_beats"] is not None and meta["acid_beats"] > 0:
                        expected = round((meta["acid_beats"] / meta["bpm"]) * 60, 4)
                        diff = round(duration - expected, 4) if duration else None

                    rows.append({
                        "filename": filepath,
                        "bpm": meta["bpm"],
                        "acid_root_note": midi_note_to_name(meta["acid_root_note"]),
                        "acid_beats": meta["acid_beats"],
                        "smpl_root_key": midi_note_to_name(meta["smpl_root_key"]),
                        "smpl_loop_start": meta["smpl_loop_start"],
                        "smpl_loop_end": meta["smpl_loop_end"],
                        "duration_sec": duration,
                        "expected_duration": expected,
                        "duration_diff": diff,
                        "other_chunks": ",".join([c for c in seen if c not in ("RIFF","WAVE","fmt ","data","acid","smpl")])
                    })

                    if not args.quiet:
                        print(f"\n=== {os.path.basename(filepath)} ===")
                        if meta["bpm"] is not None:
                            print(f"   BPM: {meta['bpm']} | Beats: {meta['acid_beats']} | Duration: {duration} sec")
                            if expected is not None:
                                print(f"   Expected: {expected} sec | Diff: {diff}")
                        else:
                            print(f"   No ACID/SMPL metadata found. Duration: {duration} sec")

                        if meta["smpl_root_key"]:
                            print(f"   Root Key (SMPL): {midi_note_to_name(meta['smpl_root_key'])}")
                        if meta["smpl_loop_start"] is not None:
                            print(f"   Loop Points: {meta['smpl_loop_start']} -> {meta['smpl_loop_end']} samples")

                        others = [c for c in seen if c not in ("RIFF","WAVE","fmt ","data","acid","smpl")]
                        if others:
                            print("   Other chunks:", ", ".join(others))

                count += 1
                if count >= args.num:
                    break
        if count >= args.num:
            break

    with open(output_csv, "w", newline="") as csvfile:
        if args.all:
            fieldnames = ["filename", "chunk", "key", "value"]
        else:
            fieldnames = ["filename", "bpm", "acid_root_note", "acid_beats", "smpl_root_key",
                          "smpl_loop_start", "smpl_loop_end", "duration_sec",
                          "expected_duration", "duration_diff", "other_chunks"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[INFO] Wrote metadata for {len(rows)} entries to {output_csv}")

if __name__ == "__main__":
    main()
