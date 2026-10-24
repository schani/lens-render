#!/usr/bin/env python3
"""Records a narrated demo video of the viewer.

Narration per segment comes from ElevenLabs (ELEVENLABS_API_KEY in .env),
cached by text; each segment lasts as long as its narration. The viewer runs
in demo mode (LENS_VIEWER_DEMO) to capture the frames, ffmpeg adds the voice.

    scripts/demo_video.py [output.mp4]
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, ".context", "demo")
FPS = 30
VOICE = "TX3LPaxmHKxFdv7VOQHJ"  # Liam: energetic
MODEL = "eleven_multilingual_v2"

# (id, on-screen title, narration, minimum seconds)
SEGMENTS = [
    ("intro", "Every ray. Every wavelength.", "This is a camera lens, simulated ray by ray, and wavelength by wavelength. Nothing you see is a filter.", 0),
    ("aperture", "Aperture", "Stop down: the bokeh shrinks, and diffraction grows star spikes around every light.", 0),
    ("focus", "Focus", "Pull focus. The blur, the colour fringes and the focus breathing all come straight out of the glass.", 0),
    ("iris", "Iris blades", "Change the iris blades, and every out-of-focus highlight takes on their shape.", 0),
    ("coating", "No anti-reflection coating", "Strip the anti-reflection coating, and internal reflections flood the frame with ghosts.", 0),
    ("summilux", "Leica Summilux-M 50 f/1.4", "Swap in a Leica Summilux fifty, wide open at f one point four: razor-thin focus, and cat's-eye bokeh towards the corners.", 0),
    ("wave", "Wave optics", "In wave-optics mode, the GPU computes thousands of diffraction-accurate point spread functions in about a second.", 0),
    ("editor", "Edit the lens", "Below the image is the lens itself. Drag an element, bend a surface, swap the glass, and the picture follows instantly.", 10),
    ("overlays", "Ghosts & focus travel", "Overlays trace the ghost reflections, and show how far the lens travels to focus close.", 0),
    ("aberrations", "Aberrations, surface by surface", "See which surface adds which aberration, and which one cancels it.", 0),
    ("pupils", "Vignetting", "See exactly which rim cuts off the light in the corners.", 0),
    ("psf", "Wavefront & PSF", "Inspect the wavefront error, and the real point spread function, across the frame.", 0),
    ("mtf", "MTF vs. Leica's published data", "And check the MTF curves against Leica's own published measurements.", 0),
    ("light", "Light loss per surface", "Down to the light lost at every single surface.", 0),
    ("outro", "lens-render", "A real lens. Simulated, not faked.", 3),
]


def api_key():
    for line in open(os.path.join(ROOT, ".env")):
        if line.startswith("ELEVENLABS_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"')
    sys.exit("ELEVENLABS_API_KEY missing in .env")


def speak(text, path):
    body = {
        "text": text,
        "model_id": MODEL,
        "voice_settings": {"stability": 0.4, "similarity_boost": 0.8, "style": 0.35, "use_speaker_boost": True, "speed": 1.08},
    }
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE}?output_format=mp3_44100_128",
        data=json.dumps(body).encode(),
        headers={"xi-api-key": api_key(), "Content-Type": "application/json", "Accept": "audio/mpeg"},
    )
    with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
        f.write(r.read())


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def main():
    out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(WORK, "lens-render-demo.mp4"))
    os.makedirs(os.path.join(WORK, "audio"), exist_ok=True)
    timeline, clips, t = [], [], 0.0
    for sid, title, text, min_dur in SEGMENTS:
        key = hashlib.sha1(f"{VOICE}|{MODEL}|{text}".encode()).hexdigest()[:10]
        path = os.path.join(WORK, "audio", f"{sid}-{key}.mp3")
        if not os.path.exists(path):
            print(f"narration: {sid}")
            speak(text, path)
        dur = max(duration(path) + 0.45, min_dur)
        timeline.append({"id": sid, "title": title, "start": t, "dur": dur})
        clips.append((path, t + 0.15))
        t += dur
    with open(os.path.join(WORK, "timeline.json"), "w") as f:
        json.dump({"fps": FPS, "segments": timeline}, f, indent=1)
    print(f"timeline: {len(timeline)} segments, {t:.1f} s")

    frames = os.path.join(WORK, "frames")
    shutil.rmtree(frames, ignore_errors=True)
    subprocess.run(["cargo", "build", "--release"], cwd=ROOT, check=True)
    env = dict(os.environ, LENS_VIEWER_DEMO=WORK)
    subprocess.run([os.path.join(ROOT, "target", "release", "lens-render"), "view", os.path.join(ROOT, "scenes", "night.toml")], cwd=ROOT, env=env, check=True)

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(frames, "%05d.jpg")]
    for path, _ in clips:
        cmd += ["-i", path]
    delays = "".join(f"[{i + 1}:a]adelay={int(s * 1000)}|{int(s * 1000)}[a{i}];" for i, (_, s) in enumerate(clips))
    mix = "".join(f"[a{i}]" for i in range(len(clips))) + f"amix=inputs={len(clips)}:normalize=0,apad[aout]"
    cmd += [
        "-filter_complex", f"[0:v]scale=1920:-2:flags=lanczos,format=yuv420p[v];{delays}{mix}",
        "-map", "[v]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart", out,
    ]
    subprocess.run(cmd, check=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
