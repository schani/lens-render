#!/usr/bin/env python3
"""Records the explainer video on lens design.

The script (explainer/script.toml) lists the chapters: narration, scene,
lens, keyframes, annotations. Narration comes from ElevenLabs with
character timings (ELEVENLABS_API_KEY in .env), cached by text in
.context/explainer/audio, so keyframes can sit on words: at = "@glow+0.2".
Writes timeline.json, records the frames (`lens-render explain`) and adds
the voice with ffmpeg.

    scripts/explainer_video.py [--only id,id] [--timeline-only] [-o out.mp4]
"""

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, ".context", "explainer")
SCRIPT = os.path.join(ROOT, "explainer", "script.toml")
VOICE = "TX3LPaxmHKxFdv7VOQHJ"
MODEL = "eleven_multilingual_v2"
# Chapter keys that aren't properties.
CHAPTER_KEYS = {"id", "title", "number", "scene", "lens", "narration", "tail", "dur", "key", "note", "tag", "tool", "tool_at", "set"}


def api_key():
    for line in open(os.path.join(ROOT, ".env")):
        if line.startswith("ELEVENLABS_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"')
    sys.exit("ELEVENLABS_API_KEY missing in .env")


ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number(s):
    """Words for a number below 100, with decimals: "2.8" → "two point eight"."""
    whole, _, frac = s.partition(".")
    n = int(whole)
    words = ONES[n] if n < 20 else TENS[n // 10] + ("-" + ONES[n % 10] if n % 10 else "")
    return words + "".join(" point " + ONES[int(d)] for d in frac)


def spoken(text):
    """The narration as it should be read: f/2.8 → "f two point eight", [5×] → 5×."""
    text = re.sub(r"\[([^\]]*)\]", r"\1", text)
    return re.sub(r"f/(\d+(?:\.\d+)?)", lambda m: "f " + number(m[1]), text)


def speak(text, settings):
    """Narration as (mp3 path, duration, alignment), cached by text and voice settings."""
    voice = {"stability": 0.5, "similarity_boost": 0.8, "style": 0.2, "use_speaker_boost": True, "speed": settings.get("speed", 1.0)}
    key = hashlib.sha1(json.dumps([VOICE, MODEL, voice, text]).encode()).hexdigest()[:12]
    base = os.path.join(WORK, "audio", key)
    if not os.path.exists(base + ".json"):
        print(f"narration: {text[:50]}…")
        req = urllib.request.Request(
            f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE}/with-timestamps?output_format=mp3_44100_128",
            data=json.dumps({"text": text, "model_id": MODEL, "voice_settings": voice}).encode(),
            headers={"xi-api-key": api_key(), "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as r:
            resp = json.load(r)
        with open(base + ".mp3", "wb") as f:
            f.write(base64.b64decode(resp["audio_base64"]))
        with open(base + ".json", "w") as f:
            json.dump(resp["alignment"], f)
    alignment = json.load(open(base + ".json"))
    return base + ".mp3", duration(base + ".mp3"), alignment


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


TIME = re.compile(r"^\s*(?:@(?P<word>[\w'’]+(?:-[^\W\d]+)*)(?:#(?P<nth>\d+))?|(?P<mark>end|speech_end|start))?\s*(?P<off>[+-]\s*\d+(?:\.\d*)?)?\s*$")


class Clock:
    """Resolves times within a chapter: seconds, "@word[#n][±s]", "end[±s]", "speech_end[±s]"."""

    def __init__(self, cid, lead, speech, alignment, dur):
        self.cid, self.lead, self.speech, self.dur = cid, lead, speech, dur
        self.chars = "".join(alignment["characters"]) if alignment else ""
        self.starts = alignment["character_start_times_seconds"] if alignment else []

    def __call__(self, spec):
        if isinstance(spec, (int, float)):
            return float(spec)
        m = TIME.match(str(spec))
        if not m:
            sys.exit(f"{self.cid}: bad time {spec!r}")
        off = float(m["off"].replace(" ", "")) if m["off"] else 0.0
        if m["word"]:
            hits = [h.start() for h in re.finditer(r"\b" + re.escape(m["word"]), self.chars, re.IGNORECASE)]
            n = int(m["nth"] or 1)
            if len(hits) < n:
                sys.exit(f"{self.cid}: word {m['word']!r} (#{n}) not in the narration")
            return self.lead + self.starts[hits[n - 1]] + off
        return {"end": self.dur, "speech_end": self.lead + self.speech, "start": 0.0, None: 0.0}[m["mark"]] + off


def path_of(entry, what):
    """A lens or bending file (path, or table with `file`) relative to the repository."""
    path = os.path.join(ROOT, entry["file"] if isinstance(entry, dict) else entry)
    if not os.path.exists(path):
        sys.exit(f"missing {what}: {path}")
    return path


def timeline(script, only):
    s = script.get("settings", {})
    lead, tail = s.get("lead", 0.35), s.get("tail", 0.8)
    chapters, clips, t = [], [], 0.0
    for ch in script["chapter"]:
        cid = ch["id"]
        text = ch.get("narration", "").strip()
        mp3, speech, align = speak(spoken(" ".join(text.split())), s) if text else (None, 0.0, None)
        dur = max(lead + speech + ch.get("tail", tail), ch.get("dur", 0.0)) if text else ch.get("dur", 3.0)
        clock = Clock(cid, lead, speech, align, dur)
        recorded = only is None or cid in only
        start = t if recorded else -1000.0
        if recorded:
            if mp3:
                clips.append((mp3, t + lead))
            t += dur
        props = {k: v for k, v in ch.items() if k not in CHAPTER_KEYS}
        props.update(ch.get("set", {}))
        for k in ("scene", "lens"):
            if k in ch:
                props[k] = ch[k]
        keys = []
        for k in ch.get("key", []):
            at = clock(k.get("at", 0))
            d = clock(k["to"]) - at if "to" in k else k.get("dur", 1.0)
            keys.append({"t": at, "dur": d, "ease": k.get("ease", "inout"), "morph": k.get("morph", False),
                         "set": {n: v for n, v in k.items() if n not in ("at", "to", "dur", "ease", "morph")}})
        keys.sort(key=lambda k: k["t"])
        notes = []
        for n in ch.get("note", []):
            at = clock(n.get("at", 0))
            until = clock(n["until"]) if "until" in n else (at + n["dur"] if "dur" in n else None)
            notes.append({**{k: v for k, v in n.items() if k not in ("at", "until", "dur")}, "t": at, "until": until})
        tags = [{"t": clock(g.get("at", 0)), "until": clock(g["until"]) if "until" in g else None, "kind": g["kind"], "text": g["text"]} for g in ch.get("tag", [])]
        tool = {"t": clock(ch.get("tool_at", "end-2.0")), "text": ch["tool"]} if "tool" in ch else None
        chapters.append({"id": cid, "title": ch.get("title", ""), "number": ch.get("number", 0), "start": start, "dur": dur,
                         "set": props, "keys": keys, "notes": notes, "tags": tags, "toolbox": tool})
    lenses = {k: {"path": path_of(v, f"lens {k}"), "label": v.get("label", k)} for k, v in script["lenses"].items()}
    scenes = {k: os.path.join(ROOT, v) for k, v in script["scenes"].items()}
    bending = path_of(s["bending"], "bending curve") if "bending" in s else None
    segments = [{"id": c["id"], "title": c["title"], "start": c["start"], "dur": c["dur"]} for c in chapters if c["start"] >= 0]
    out = {"fps": s.get("fps", 30), "jpeg_quality": 92, "width": s.get("render_width", 1800), "segments": segments, "chapters": chapters,
           "lenses": lenses, "scenes": scenes, "bending": bending, "recap": script.get("recap", [])}
    return out, clips, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated chapter ids to record")
    ap.add_argument("--timeline-only", action="store_true", help="write timeline.json and stop")
    ap.add_argument("-o", "--out", default=os.path.join(WORK, "lens-design-explainer.mp4"))
    args = ap.parse_args()
    os.makedirs(os.path.join(WORK, "audio"), exist_ok=True)
    script = tomllib.load(open(SCRIPT, "rb"))
    only = set(args.only.split(",")) if args.only else None
    tl, clips, total = timeline(script, only)
    with open(os.path.join(WORK, "timeline.json"), "w") as f:
        json.dump(tl, f, indent=1)
    print(f"timeline: {len(tl['segments'])} chapters, {total:.1f} s")
    for c in tl["chapters"]:
        if c["start"] >= 0:
            print(f"  {c['start']:6.1f}  {c['dur']:5.1f}  {c['id']}")
    if args.timeline_only:
        return

    frames = os.path.join(WORK, "frames")
    shutil.rmtree(frames, ignore_errors=True)
    subprocess.run(["cargo", "build", "--release"], cwd=ROOT, check=True)
    subprocess.run([os.path.join(ROOT, "target", "release", "lens-render"), "explain", WORK], cwd=ROOT, check=True)

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(tl["fps"]), "-i", os.path.join(frames, "%05d.jpg")]
    for path, _ in clips:
        cmd += ["-i", path]
    delays = "".join(f"[{i + 1}:a]adelay={int(s * 1000)}|{int(s * 1000)}[a{i}];" for i, (_, s) in enumerate(clips))
    mix = "".join(f"[a{i}]" for i in range(len(clips))) + f"amix=inputs={len(clips)}:normalize=0,apad[aout]"
    cmd += [
        "-filter_complex", f"[0:v]scale=1920:-2:flags=lanczos,format=yuv420p[v];{delays}{mix}",
        "-map", "[v]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart", os.path.abspath(args.out),
    ]
    subprocess.run(cmd, check=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
