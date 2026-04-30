r"""
Transcribe interview audio files (*itw*.m4a) using WhisperX + pyannote diarization.

════════════════════════════════════════════════════════════════════════════════
CONTEXT  (written on Fedora, to be implemented on the Windows machine)
════════════════════════════════════════════════════════════════════════════════

WHY THIS SCRIPT EXISTS
----------------------
Each participant folder (HITLS/P0x/) contains one or more interview recordings:
    P02_itw.m4a
    P04_itw_tarc.m4a        ← some participants have a second TARC interview
    P06_itw_part_1.m4a      ← some are split across multiple files
    etc.
This script batch-transcribes all of them, with speaker labels, so that a
theory-driven thematic analysis can be performed downstream (see thematic_analysis.py).

PLATFORM RECOMMENDATION
-----------------------
Run on the Windows PC with the RTX 4500A (20 GB VRAM), NOT on Fedora CPU.
Reason: WhisperX with large-v3-turbo runs ~60-100× realtime on CUDA vs ~8× on CPU.
The 20 GB VRAM easily fits large-v3-turbo (large-v3 also fits).

════════════════════════════════════════════════════════════════════════════════
WINDOWS SETUP — do this before running the script
════════════════════════════════════════════════════════════════════════════════

# 1. Create a virtual environment (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install PyTorch with CUDA 12.x support first (check https://pytorch.org)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install WhisperX (pulls faster-whisper, pyannote, transformers automatically)
pip install whisperx

# 4. ffmpeg must be on PATH — download from https://ffmpeg.org/download.html
#    and add its bin/ directory to the Windows PATH environment variable.
#    WhisperX calls ffmpeg internally to decode .m4a files.

# 5. HuggingFace token (needed for pyannote diarization models)
#    a. Create a free account at https://huggingface.co
#    b. Generate an access token at https://huggingface.co/settings/tokens
#    c. Accept the license for BOTH models (click "Agree" on the model page):
#         https://huggingface.co/pyannote/speaker-diarization-3.1
#         https://huggingface.co/pyannote/segmentation-3.0
#    d. Either set the env variable:   set HF_TOKEN=hf_xxxxxxxxxxxx
#       or pass it at runtime:         python transcription.py --hf-token hf_xxx ...

════════════════════════════════════════════════════════════════════════════════
HOW WHISPERX + PYANNOTE WORK TOGETHER
════════════════════════════════════════════════════════════════════════════════

WhisperX alone gives transcription with word-level timestamps but NO speaker info.
Pyannote speaker-diarization-3.1 gives speaker turn segments (SPEAKER_00, SPEAKER_01)
but NO words.

WhisperX's built-in pipeline chains them:
    1. whisperx.load_model(...)       → Whisper large-v3-turbo, CUDA
    2. model.transcribe(audio)        → segments with word timestamps
    3. whisperx.load_align_model(...) → forced alignment for precise word offsets
    4. whisperx.align(...)            → word-level alignment
    5. whisperx.DiarizationPipeline(...)  → pyannote, detects speaker turns
    6. whisperx.assign_word_speakers(diarize_segments, result)
                                      → each word gets a speaker label

In a 2-person interview you will get SPEAKER_00 and SPEAKER_01.
Whisper does NOT know which is the interviewer and which is the interviewee —
you will need to check the first few lines of the transcript and rename them
(see the --interviewer-id flag below, or edit the JSON manually).

════════════════════════════════════════════════════════════════════════════════
OUTPUTS  (per audio file)
════════════════════════════════════════════════════════════════════════════════

For each *itw*.m4a found the script writes two files next to the audio:

  P0x/<stem>_transcript.json
    Full structured data: list of segments, each with:
      { "start": float, "end": float, "speaker": "INTERVIEWER"|"PARTICIPANT"|"SPEAKER_XX",
        "text": str, "words": [ {"word": str, "start": float, "end": float} ] }
    This JSON is the input for thematic_analysis.py.

  P0x/<stem>_transcript.txt
    Human-readable version:
      [00:01:23 → 00:01:45]  PARTICIPANT
      That's when I started to trust the system more, because it had already...

If the output files already exist the script skips that audio (use --force to redo).

════════════════════════════════════════════════════════════════════════════════
DOWNSTREAM: thematic_analysis.py  (to be written after transcription)
════════════════════════════════════════════════════════════════════════════════

Once you have the *_transcript.json files, the next script will:

  A. LLM pre-coding (local, privacy-safe)
     - Load a codebook.yaml file you define (code name + definition + examples)
     - Call a local Ollama LLM (llama3.3:70b-instruct-q4 or mistral-large)
       running on the same Windows machine via   ollama serve
       API endpoint: http://localhost:11434/api/chat
     - For each PARTICIPANT segment: ask the LLM to assign 0-N codes + justification
     - Output: P0x/<stem>_coded.json

  B. Keyword-in-context (KWIC) search
     - Given a search term, return all segments containing it with context window
     - Useful for cross-participant quote retrieval during write-up

  C. Semantic similarity retrieval
     - Embed code definitions + all segments with sentence-transformers
       (all-MiniLM-L6-v2 or paraphrase-multilingual if French interviews)
     - Rank segments by cosine similarity to a code → recall check

  D. Analysis & visualisation
     - Code frequency heatmap (participants × codes) → plots/code_heatmap.png
     - Code co-occurrence network → plots/cooccurrence_network.png
     - Annotated HTML report: excerpts per code, all participants, with timestamps

  codebook.yaml structure (you define this based on your theory):
    codes:
      - name: trust_calibration
        definition: "Participant expresses adjustment of trust level toward automation"
        examples:
          - "I started trusting it more after that"
          - "I wasn't sure it would handle that situation"
      - name: workload_management
        definition: "..."

════════════════════════════════════════════════════════════════════════════════
USAGE
════════════════════════════════════════════════════════════════════════════════

    python transcription.py [participant_dirs...]  [options]

    python transcription.py HITLS/P02
    python transcription.py HITLS/P02 HITLS/P03 HITLS/P06
    python transcription.py                          # interactive: all P0x dirs

    Options:
      --model MODEL          WhisperX model name (default: large-v3-turbo)
                             Choices: tiny, base, small, medium, large-v2,
                                      large-v3, large-v3-turbo
      --language LANG        ISO 639-1 code, e.g. fr, en (default: auto-detect)
      --hf-token TOKEN       HuggingFace token for pyannote (or set HF_TOKEN env)
      --no-diarize           Skip speaker diarization (faster, no speaker labels)
      --interviewer-id ID    Rename SPEAKER_XX to INTERVIEWER in output
                             e.g. --interviewer-id SPEAKER_00
      --force                Re-transcribe even if output files already exist
      --device DEVICE        torch device: cuda (default) or cpu

════════════════════════════════════════════════════════════════════════════════
IMPLEMENTATION NOTES  (for the person coding this on Windows)
════════════════════════════════════════════════════════════════════════════════

Key WhisperX API calls (v3.x — verify against installed version):

    import whisperx, torch

    device = "cuda"
    audio_file = "P02/P02_itw.m4a"
    batch_size = 16          # reduce if VRAM OOM; 16 is safe for 4500A + large-v3-turbo
    compute_type = "float16" # or "int8" for even less VRAM

    model = whisperx.load_model("large-v3-turbo", device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size, language="fr")

    # Word-level alignment
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], align_model, metadata, audio, device)

    # Diarization
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=HF_TOKEN, device=device)
    diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=2)

    # Assign speakers to words/segments
    result = whisperx.assign_word_speakers(diarize_segments, result)
    # result["segments"] now has a "speaker" key on each segment

The script should free GPU memory between participants:
    del model; torch.cuda.empty_cache()  # after transcription
    del align_model; torch.cuda.empty_cache()  # after alignment

Speaker renaming:
    After assign_word_speakers, iterate result["segments"] and replace
    the speaker string:  "SPEAKER_00" → "INTERVIEWER"  (if --interviewer-id given)
    The remaining speaker becomes "PARTICIPANT".

TXT format helper:
    def fmt_time(seconds: float) -> str:
        h, r = divmod(int(seconds), 3600)
        m, s = divmod(r, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    Then:  f"[{fmt_time(seg['start'])} → {fmt_time(seg['end'])}]  {seg['speaker']}\n{seg['text'].strip()}\n"

JSON output: use json.dump(result["segments"], f, ensure_ascii=False, indent=2)
  Add a top-level wrapper:
    {
      "participant": "P02",
      "audio_file": "P02_itw.m4a",
      "language": "fr",
      "model": "large-v3-turbo",
      "transcribed_at": "2026-...",
      "segments": [...]
    }

════════════════════════════════════════════════════════════════════════════════
TODO — implement the sections below
════════════════════════════════════════════════════════════════════════════════
"""

# ── IMPORTS ───────────────────────────────────────────────────────────────────
import whisperx
import torch
import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL   = "large-v3-turbo"
DEFAULT_DEVICE  = "cuda"
DEFAULT_COMPUTE = "float16"   # use "int8" if VRAM is tight
BATCH_SIZE      = 16          # safe for RTX 4500A + large-v3-turbo; lower if OOM


# ── HELPERS ───────────────────────────────────────────────────────────────────

def fmt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def find_participant_dirs(base: Path) -> list[Path]:
    """Return sorted list of P0x directories under base."""
    return sorted(p for p in base.iterdir() if p.is_dir() and re.match(r"P\d+$", p.name))


def find_audio_files(participant_dir: Path) -> list[Path]:
    """Return all *itw*.m4a files in a participant directory."""
    return sorted(participant_dir.glob("*itw*.m4a"))


def rename_speakers(segments: list[dict], interviewer_id: str | None) -> list[dict]:
    """
    Replace raw SPEAKER_XX labels with INTERVIEWER / PARTICIPANT.
    If interviewer_id is None, labels are left as-is (SPEAKER_00, SPEAKER_01, …).
    """
    if not interviewer_id:
        return segments
    for seg in segments:
        raw = seg.get("speaker", "")
        if raw == interviewer_id:
            seg["speaker"] = "INTERVIEWER"
        elif raw:
            seg["speaker"] = "PARTICIPANT"
    return segments


def write_txt(segments: list[dict], out_path: Path) -> None:
    """Write human-readable transcript with speaker labels and timestamps."""
    with out_path.open("w", encoding="utf-8") as fh:
        current_speaker = None
        for seg in segments:
            speaker = seg.get("speaker", "UNKNOWN")
            start   = fmt_time(seg.get("start", 0))
            end     = fmt_time(seg.get("end", 0))
            text    = seg.get("text", "").strip()
            if not text:
                continue
            if speaker != current_speaker:
                fh.write(f"\n[{start} → {end}]  {speaker}\n")
                current_speaker = speaker
            fh.write(f"{text}\n")


def write_json(segments: list[dict], audio_path: Path,
               language: str, model_name: str, out_path: Path) -> None:
    """Write structured JSON transcript."""
    participant = audio_path.parent.name   # e.g. "P02"
    payload = {
        "participant":    participant,
        "audio_file":     audio_path.name,
        "language":       language,
        "model":          model_name,
        "transcribed_at": datetime.now(timezone.utc).isoformat(),
        "segments":       segments,
    }
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


# ── CORE TRANSCRIPTION ────────────────────────────────────────────────────────

def transcribe_file(
    audio_path: Path,
    model_name: str,
    language: str | None,
    hf_token: str,
    device: str,
    compute_type: str,
    diarize: bool,
    interviewer_id: str | None,
    force: bool,
) -> None:
    """Transcribe a single audio file and write .json + .txt outputs."""

    stem      = audio_path.stem                        # e.g. P02_itw
    json_out  = audio_path.parent / f"{stem}_transcript.json"
    txt_out   = audio_path.parent / f"{stem}_transcript.txt"

    if not force and json_out.exists() and txt_out.exists():
        print(f"  [SKIP] {audio_path.name} — outputs already exist (use --force to redo)")
        return

    print(f"  [TRANSCRIBE] {audio_path.name}")

    # ── 1. Load model & transcribe ───────────────────────────────────────────
    model = whisperx.load_model(model_name, device, compute_type=compute_type)
    audio = whisperx.load_audio(str(audio_path))
    transcribe_kwargs = {"batch_size": BATCH_SIZE}
    if language:
        transcribe_kwargs["language"] = language
    result = model.transcribe(audio, **transcribe_kwargs)
    detected_language = result["language"]
    print(f"    Detected language: {detected_language}")
    del model; torch.cuda.empty_cache()

    # ── 2. Word-level alignment ───────────────────────────────────────────────
    align_model, metadata = whisperx.load_align_model(
        language_code=detected_language, device=device)
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, device,
        return_char_alignments=False)
    del align_model; torch.cuda.empty_cache()

    # ── 3. Diarization ───────────────────────────────────────────────────────
    if diarize:
        diarize_model = whisperx.diarize.DiarizationPipeline(
            token=hf_token, device=device)
        diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=2)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        del diarize_model; torch.cuda.empty_cache()

    # ── 4. Speaker renaming ───────────────────────────────────────────────────
    segments = rename_speakers(result["segments"], interviewer_id)

    # ── 5. Write outputs ─────────────────────────────────────────────────────
    write_json(segments, audio_path, detected_language, model_name, json_out)
    write_txt(segments, txt_out)
    print(f"    → {json_out.name}")
    print(f"    → {txt_out.name}")


# ── PARTICIPANT-LEVEL ENTRY POINT ─────────────────────────────────────────────

def process_participant(participant_dir: Path, args: argparse.Namespace) -> None:
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    if args.diarize and not hf_token:
        print("  [WARN] No HuggingFace token — diarization disabled for this run.")
        print("         Set HF_TOKEN env var or pass --hf-token.")

    audio_files = find_audio_files(participant_dir)
    if not audio_files:
        print(f"  [SKIP] {participant_dir.name} — no *itw*.m4a files found")
        return

    print(f"\n── {participant_dir.name} ({len(audio_files)} file(s)) ──")
    for audio in audio_files:
        transcribe_file(
            audio_path     = audio,
            model_name     = args.model,
            language       = args.language,
            hf_token       = hf_token,
            device         = args.device,
            compute_type   = DEFAULT_COMPUTE,
            diarize        = args.diarize and bool(hf_token),
            interviewer_id = args.interviewer_id,
            force          = args.force,
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Transcribe interview audio files using WhisperX + pyannote diarization."
    )
    p.add_argument(
        "participant_dirs", nargs="*",
        help="One or more P0x directories. If omitted, all P0x dirs under HITLS/ are used."
    )
    p.add_argument(
        "--model", default=DEFAULT_MODEL,
        choices=["tiny", "base", "small", "medium",
                 "large-v2", "large-v3", "large-v3-turbo"],
        help=f"WhisperX model (default: {DEFAULT_MODEL})"
    )
    p.add_argument(
        "--language", default=None,
        help="ISO 639-1 language code, e.g. 'fr' or 'en'. Default: auto-detect."
    )
    p.add_argument(
        "--hf-token", dest="hf_token", default=None,
        help="HuggingFace access token for pyannote diarization models."
    )
    p.add_argument(
        "--no-diarize", dest="diarize", action="store_false", default=True,
        help="Skip speaker diarization (no HF token needed, faster)."
    )
    p.add_argument(
        "--interviewer-id", dest="interviewer_id", default=None,
        metavar="SPEAKER_XX",
        help="Raw speaker label to rename to INTERVIEWER (e.g. SPEAKER_00)."
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-transcribe even if output files already exist."
    )
    p.add_argument(
        "--device", default=DEFAULT_DEVICE, choices=["cuda", "cpu"],
        help=f"PyTorch device (default: {DEFAULT_DEVICE})."
    )
    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # Resolve the base HITLS directory relative to this script
    base = Path(__file__).parent

    if args.participant_dirs:
        dirs = [Path(d) for d in args.participant_dirs]
    else:
        dirs = find_participant_dirs(base)
        if not dirs:
            print(f"No P0x directories found under {base}")
            sys.exit(1)
        print("Participants found:")
        for d in dirs:
            print(f"  {d}")
        ans = input("Transcribe all? [Y/n] ").strip().lower()
        if ans not in ("", "y", "yes"):
            sys.exit(0)

    for d in dirs:
        if not d.exists():
            print(f"[ERROR] Directory not found: {d}")
            continue
        process_participant(d, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
