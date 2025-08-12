# -*- coding: utf-8 -*-

import queue, threading, tkinter as tk
import wave, subprocess, os, signal
from pathlib import Path
from tkinter.scrolledtext import ScrolledText

import numpy as np
import sounddevice as sd
import torch
from transformers import (
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniProcessor,
)

# ---------- Load Qwen2.5-Omni (takes a few minutes in the first time)  ----------
model_id = "Qwen/Qwen2.5-Omni-7B"
print(f"🔄 Loading {model_id}...")
model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)
processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
print("✅ Model ready")

RATE    = 16_000
WAVPATH = Path("/tmp/record.wav")

class Recorder:
    def __init__(self):
        self.q = queue.Queue()
        self.frames, self.running, self.stream = [], False, None
        self.proc = None

    def _cb(self, indata, frames, time, status):
        if status: print("⚠️", status)
        self.q.put(indata.copy())

    def start(self):
        cmd = [
          "ffmpeg", "-y", "-f", "pulse", "-i", "default",
          "-ac", "1", "-ar", "16000",
          str(WAVPATH)
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)

    def stop(self) -> Path:
        if self.proc:
            # Ctrl-C
            self.proc.send_signal(signal.SIGINT)
            self.proc.wait()
  
        return WAVPATH

def qwen_transcribe(path: Path) -> str:
    conv = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are Qwen, a virtual human developed by the Qwen Team, "
                            "capable of perceiving auditory and visual inputs and "
                            "generating text responses.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "path": str(path)},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        conv,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        use_audio_in_video=True,
    ).to(model.device)

    ids = model.generate(**inputs, use_audio_in_video=True, max_new_tokens=256)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]

# ---------- Tkinter GUI ----------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Qwen2.5-Omni Local Voice")
        self.geometry("640x400")

        self.btn = tk.Button(self, text="Record", width=12, command=self.toggle)
        self.btn.pack(pady=10)
        self.btn_default_bg = self.btn.cget("background")

        self.txt = ScrolledText(self, wrap=tk.WORD, font=("Consolas", 10))
        self.txt.pack(expand=True, fill="both")
        self.txt.insert(tk.END, "Press the button to record.")

        self.rec = Recorder()
        self.recording = False

    def toggle(self):
        if not self.recording:
            # --- start recording ---
            self.recording = True
            self.btn.config(text="■ Stop", bg="red")
            self.rec.start()
            self.txt.insert(tk.END, "▶  Recording...\n"); self.txt.see(tk.END)
        else:
            # --- finish recording ---
            self.recording = False
            self.btn.config(text="Record", bg=self.btn_default_bg)
            self.txt.insert(tk.END, "■ Finished recording. The model is making an inference...\n"); self.txt.see(tk.END)
            wav = self.rec.stop()
            threading.Thread(target=self._infer, args=(wav,), daemon=True).start()

    def _infer(self, wav: Path):
        try:
            res = qwen_transcribe(wav)
            self.txt.insert(tk.END, f"Qwen: {res}\n\n")
        except Exception as e:
            self.txt.insert(tk.END, f"!!Error!!: {e}\n\n")
        self.txt.see(tk.END)

if __name__ == "__main__":
    App().mainloop()

