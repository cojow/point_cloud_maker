"""
view_log.py - a lightweight viewer for this project's .out job logs.

These logs mix a handful of genuinely useful lines (stage headers, warnings,
errors, summary stats) into thousands of near-identical progress lines
(EXIF extraction, feature matching, depthmap computation, one per image) that
make the file painful to read top to bottom and too big for a plain text
editor to open comfortably. This collapses consecutive repeats of the same
*kind* of line down to a first/last/count summary, colorizes the lines that
actually matter, and can follow a job that's still running - stdlib only, no
dependencies, meant to be piped into `less -R` for normal terminal scrolling
or run with -f like `tail -f`.

Usage:
    python view_log.py path/to/job.out            # view once, collapsed
    python view_log.py path/to/job.out --raw       # view once, uncollapsed
    python view_log.py path/to/job.out -f          # follow a running job
    python view_log.py path/to/job.out | less -R   # scroll interactively
"""
import argparse
import re
import sys
import time

RESET = "\033[0m"
COLORS = {
    "stage":   "\033[1;36m",   # bold cyan  - "--- ... ---" / "[N/10]" section headers
    "warn":    "\033[1;33m",   # bold yellow - "[!]" warnings
    "error":   "\033[1;31m",   # bold red   - failures, kills, tracebacks
    "success": "\033[1;32m",   # bold green - completion / PASS lines
    "dim":     "\033[2m",      # dim        - the "... N similar lines ..." marker
}

STAGE_RE = re.compile(r'^(---.*---|\[\d+/\d+\]|=== .* ===)')
WARN_RE = re.compile(r'\[!\]')
# Deliberately NOT matching a bare "FAILED": OpenSfM's own per-pair matching
# DEBUG output ends every unsuccessful pair with "Matches: FAILED" - a normal,
# expected result (most image pairs in a flight don't overlap), not a
# problem, and there are tens of thousands of these lines in a typical run.
# Matching it here would mark that entire, genuinely-collapsible block as
# "errors" and defeat the collapsing this tool exists to do.
ERROR_RE = re.compile(r'\b(Error|ERROR|Killed|oom_kill|Traceback|Exception)\b')
SUCCESS_RE = re.compile(r'\b(Pipeline Complete|PASS|Cropped \d)\b')

# Collapses consecutive "chatter" - per-image progress lines (EXIF
# extraction, feature reading/matching, depthmap compute/clean/prune) that
# are the real bulk of these files. Classified by SHAPE (has a timestamp AND
# an image filename, and isn't a stage/warn/error/success line), not by exact
# text match - several different chatter messages get interleaved by the
# container's own worker pool (e.g. "Reading data for X" / "Extracting
# features for Y" alternate line by line), so no two consecutive lines are
# ever textually identical even though the whole block is still just noise.
_TS_RE = re.compile(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}')
_IMG_RE = re.compile(r'\b[\w.-]+\.(?:jpg|jpeg|tif|tiff)(?:\.jpg)?\b', re.IGNORECASE)
# A few per-call DEBUG lines (e.g. feature detection's "Found N points in Ts")
# carry no filename at all, only numbers - still one-per-image bulk chatter,
# just a shape _IMG_RE alone can't catch.
_BULK_DEBUG_RE = re.compile(r'Found \d+ points in [\d.]+s')
COLLAPSE_THRESHOLD = 4  # a run shorter than this is just shown in full


def is_chatter(line):
    if STAGE_RE.match(line.strip()) or WARN_RE.search(line) or ERROR_RE.search(line) or SUCCESS_RE.search(line):
        return False
    if not _TS_RE.search(line):
        return False
    return bool(_IMG_RE.search(line)) or bool(_BULK_DEBUG_RE.search(line))


def colorize(line):
    if ERROR_RE.search(line):
        return COLORS["error"] + line + RESET
    if WARN_RE.search(line):
        return COLORS["warn"] + line + RESET
    if SUCCESS_RE.search(line):
        return COLORS["success"] + line + RESET
    if STAGE_RE.match(line.strip()):
        return COLORS["stage"] + line + RESET
    return line


def emit_run(out, run, chatter):
    """Prints one run of consecutive lines that are all chatter, or all not:
    non-chatter is always shown in full; a chatter run collapses to its
    first line, a count of what's between, and its last line (which for
    depthmap/matching stages usually carries the batch's final tally)."""
    if not chatter or len(run) < COLLAPSE_THRESHOLD:
        for line in run:
            out.write(colorize(line) + "\n")
    else:
        out.write(colorize(run[0]) + "\n")
        out.write(f"{COLORS['dim']}      ... {len(run) - 2:,} similar line(s) omitted ...{RESET}\n")
        out.write(colorize(run[-1]) + "\n")
    out.flush()


def process(lines, out, collapse=True):
    if not collapse:
        for line in lines:
            out.write(colorize(line) + "\n")
        return
    run, run_is_chatter = [], None
    for line in lines:
        chatter = is_chatter(line)
        if chatter == run_is_chatter:
            run.append(line)
            continue
        if run:
            emit_run(out, run, run_is_chatter)
        run, run_is_chatter = [line], chatter
    if run:
        emit_run(out, run, run_is_chatter)


def follow(path, collapse):
    """Tails the file like `tail -f`, applying the same collapsing/coloring
    to each new line as it's written - for watching a job that's still
    running instead of waiting for it to finish."""
    with open(path, 'r', errors='replace') as f:
        f.seek(0, 2)  # start at end - only show new lines from here
        run, run_is_chatter = [], None
        try:
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.5)
                    continue
                line = line.rstrip("\n")
                if not collapse:
                    sys.stdout.write(colorize(line) + "\n")
                    sys.stdout.flush()
                    continue
                chatter = is_chatter(line)
                if chatter == run_is_chatter:
                    run.append(line)
                    continue
                if run:
                    emit_run(sys.stdout, run, run_is_chatter)
                run, run_is_chatter = [line], chatter
        except KeyboardInterrupt:
            if run:
                emit_run(sys.stdout, run, run_is_chatter)


def main():
    parser = argparse.ArgumentParser(description="Lightweight viewer for this project's .out job logs.")
    parser.add_argument("path", help="Path to the .out file")
    parser.add_argument("--raw", action="store_true", help="Show every line uncollapsed")
    parser.add_argument("-f", "--follow", action="store_true", help="Follow a running job, like tail -f")
    args = parser.parse_args()

    if args.follow:
        follow(args.path, collapse=not args.raw)
        return

    with open(args.path, 'r', errors='replace') as f:
        lines = [line.rstrip("\n") for line in f]
    process(lines, sys.stdout, collapse=not args.raw)


if __name__ == "__main__":
    main()
