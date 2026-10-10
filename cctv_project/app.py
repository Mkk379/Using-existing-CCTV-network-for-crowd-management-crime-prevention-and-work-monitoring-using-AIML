"""CCTV AI: Crowd Management, Crime Prevention & Work Monitoring (GUI).
Run:  python app.py
"""
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import END, filedialog, messagebox, simpledialog, ttk

import cv2
from PIL import Image, ImageTk

from cctv_ai.analytics import Zone, load_zones, save_zones
from cctv_ai.config import Settings
from cctv_ai.detectors import PersonDetector, WeaponDetector
from cctv_ai.pipeline import Analyzer, StreamProcessor, analyze_image, fit_width

os.chdir(Path(__file__).resolve().parent)  # make relative model/output paths work from anywhere
VIEW_W, VIEW_H = 760, 520
LEVEL_FG = {"LOW": "#2e9e44", "MEDIUM": "#d4a017", "HIGH": "#e67e22", "CRITICAL": "#c0392b"}


class App:
    def __init__(self, root):
        self.root = root
        self.settings = Settings.load()
        self.person_det = self.weapon_det = None
        self.proc = None
        self.q = queue.Queue()
        self.latest = None          # newest annotated frame from the worker thread
        self._photo = None
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.exit_app)
        self.poll()

    # ------------------------------------------------------------ UI
    def _build_ui(self):
        r = self.root
        r.title("CCTV AI: Crowd Management & Crime Prevention")
        r.configure(bg="#f0f4f8")
        w, h = 1300, 820
        r.geometry(f"{w}x{h}+{(r.winfo_screenwidth() - w) // 2}+{(r.winfo_screenheight() - h) // 2}")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Helvetica", 11, "bold"), padding=8, background="#4a90e2", foreground="white")
        style.map("TButton", background=[("active", "#357abd")])
        style.configure("TLabel", font=("Helvetica", 14), background="#f0f4f8", foreground="#333")
        style.configure("Header.TLabel", font=("Helvetica", 18, "bold"), background="#2c3e50", foreground="white")
        style.configure("Header.TFrame", background="#2c3e50")
        style.configure("TFrame", background="#f0f4f8")
        style.configure("TLabelframe", background="#f0f4f8")
        style.configure("TLabelframe.Label", background="#f0f4f8", font=("Helvetica", 11, "bold"))
        style.configure("TCheckbutton", background="#f0f4f8", font=("Helvetica", 11))

        hf = ttk.Frame(r, padding=10, style="Header.TFrame")
        hf.pack(fill="x")
        ttk.Label(hf, text="CCTV AI: Crowd Management, Crime Prevention & Work Monitoring",
                  style="Header.TLabel").pack(pady=6)

        content = ttk.Frame(r, padding=10)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=0)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        # Left: live view
        vf = ttk.LabelFrame(content, text="Live View", padding=6)
        vf.grid(row=0, column=0, sticky="n", padx=(0, 10))
        holder = tk.Frame(vf, width=VIEW_W, height=VIEW_H, bg="black")
        holder.pack_propagate(False)
        holder.pack()
        self.video = tk.Label(holder, bg="black", fg="#aaa", text="No source running", font=("Arial", 14))
        self.video.pack(fill="both", expand=True)

        # Right: analytics + log
        right = ttk.Frame(content)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        sf = ttk.LabelFrame(right, text="Live Analytics", padding=8)
        sf.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.v_people, self.v_level = tk.StringVar(value="-"), tk.StringVar(value="-")
        self.v_weap, self.v_zones = tk.StringVar(value="-"), tk.StringVar(value="No zones defined")
        for i, (name, var) in enumerate([("People", self.v_people), ("Crowd level", self.v_level),
                                         ("Weapons", self.v_weap), ("Zones", self.v_zones)]):
            tk.Label(sf, text=name + ":", font=("Arial", 12, "bold"), bg="#f0f4f8", anchor="w").grid(row=i, column=0, sticky="w")
            lab = tk.Label(sf, textvariable=var, font=("Arial", 12), bg="#f0f4f8", anchor="w", justify="left", wraplength=360)
            lab.grid(row=i, column=1, sticky="w", padx=8)
            if name == "Crowd level":
                self.lbl_level = lab

        lf = ttk.LabelFrame(right, text="Output Log", padding=6)
        lf.grid(row=1, column=0, sticky="nsew")
        self.text = tk.Text(lf, height=15, width=50, font=("Arial", 10), bg="white", fg="#333", bd=2, relief="sunken")
        sb = ttk.Scrollbar(lf, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.text.pack(fill="both", expand=True)
        self.text.tag_config("alert", foreground="#c0392b", font=("Arial", 10, "bold"))

        # Buttons
        bf = ttk.Frame(r, padding=10)
        bf.pack(fill="x")
        buttons = [
            ("Load Models", self.load_models), ("Image Detection", self.image_detection),
            ("Video File", self.video_detection), ("Webcam", self.webcam),
            ("IP / RTSP Camera", self.ip_camera), ("Stop", self.stop_stream),
            ("Add Zone", self.add_zone), ("Clear Zones", self.clear_zones),
            ("Crowd Graph", self.graph), ("Exit", self.exit_app),
        ]
        for i, (label, cmd) in enumerate(buttons):
            ttk.Button(bf, text=label, command=cmd).grid(row=i // 5, column=i % 5, padx=6, pady=4, sticky="ew")
            bf.columnconfigure(i % 5, weight=1)
        self.heat_var = tk.BooleanVar(value=self.settings.show_heatmap)
        ttk.Checkbutton(bf, text="Show density heatmap", variable=self.heat_var,
                        command=lambda: setattr(self.settings, "show_heatmap", self.heat_var.get())
                        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=6)

        self.status = tk.StringVar(value="Ready: click 'Load Models' first")
        ttk.Label(r, textvariable=self.status, font=("Arial", 10), background="#d9e2ec", relief="sunken",
                  anchor="w", padding=5).pack(fill="x", side="bottom")

    # ------------------------------------------------------------ helpers
    def log(self, msg, tag=None):
        self.text.insert(END, msg + "\n", tag)
        self.text.see(END)

    def set_status(self, msg):
        self.status.set(msg)

    def poll(self):
        """Main-thread pump: the worker threads never touch Tk directly."""
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self.log(payload)
                elif kind == "alert":
                    self.log("⚠ " + payload, "alert")
                    self.root.bell()
                elif kind == "stats":
                    self.update_stats(payload)
                elif kind == "models":
                    self.person_det, self.weapon_det = payload
                    ok = self.person_det is not None
                    self.set_status("Models loaded" if ok else "Error loading person model")
                elif kind == "done":
                    self.set_status("Ready")
        except queue.Empty:
            pass
        if self.latest is not None:
            frame, self.latest = self.latest, None
            self.show(frame)
        self.root.after(30, self.poll)

    def show(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        s = min(VIEW_W / w, VIEW_H / h)
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)))
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video.configure(image=self._photo, text="")

    def update_stats(self, st):
        self.v_people.set(f"{st['people']}   (peak {st['peak']}, avg {st['avg']:.1f})")
        self.v_level.set(st["level"])
        self.lbl_level.configure(fg=LEVEL_FG[st["level"]])
        self.v_weap.set(", ".join(st["weapons"]) if st["weapons"] else "none")
        if st["zones"]:
            self.v_zones.set("\n".join(f"{n} ({k}): {c} inside" for n, k, c in st["zones"]))
        else:
            self.v_zones.set("No zones defined")

    def ready(self):
        if self.person_det is None:
            messagebox.showwarning("Models not loaded", "Click 'Load Models' first.")
            return False
        return True

    def make_analyzer(self):
        return Analyzer(self.settings, self.person_det, self.weapon_det,
                        load_zones(self.settings.zones_file),
                        on_alert=lambda m: self.q.put(("alert", m)))

    # ------------------------------------------------------------ actions
    def load_models(self):
        self.text.delete("1.0", END)
        self.set_status("Loading models (first run may download files)...")
        threading.Thread(target=self._load_worker, daemon=True).start()

    def _load_worker(self):
        person = weapon = None
        try:
            person = PersonDetector(self.settings.person_model_path, self.settings.person_conf)
            self.q.put(("log", f"Person detection model loaded: {person.backend}"))
        except Exception as e:
            self.q.put(("log", f"Error loading person model: {e}"))
        try:
            weapon = WeaponDetector(self.settings.weapon_model_path, self.settings.weapon_conf)
            self.q.put(("log", f"Weapon detection model loaded. Classes: {list(weapon.names.values())}"))
        except Exception as e:
            self.q.put(("log", f"Weapon model not loaded ({e}). Weapon detection is disabled."))
        self.q.put(("models", (person, weapon)))

    def image_detection(self):
        if not self.ready():
            return
        self.stop_stream()
        path = filedialog.askopenfilename(initialdir="images", title="Select Image",
                                          filetypes=(("Images", "*.jpg *.jpeg *.png"), ("All files", "*.*")))
        if not path:
            return
        self.set_status("Processing image...")
        out, st = analyze_image(path, self.make_analyzer())
        if out is None:
            self.log("Failed to load image.")
            return
        self.show(out)
        self.update_stats(st)
        save = Path(self.settings.output_dir) / f"annotated_{Path(path).name}"
        save.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save), out)
        self.log(f"{Path(path).name}: {st['people']} people, crowd {st['level']}, weapons: {st['weapons'] or 'none'}")
        self.log(f"Annotated image saved to {save}")
        self.set_status("Image processing complete")

    def video_detection(self):
        if not self.ready():
            return
        path = filedialog.askopenfilename(initialdir="Video", title="Select Video",
                                          filetypes=(("Video", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*")))
        if path:
            self.start_stream(path, Path(path).name)

    def webcam(self):
        if self.ready():
            self.start_stream(0, "webcam 0")

    def ip_camera(self):
        if not self.ready():
            return
        url = simpledialog.askstring("IP / RTSP camera", "Stream URL (e.g. rtsp://user:pass@192.168.1.10:554/stream1):")
        if url:
            self.start_stream(url.strip(), "IP camera")

    def start_stream(self, source, label):
        self.stop_stream()
        self.text.delete("1.0", END)
        self.log(f"Starting: {label}   (click 'Stop' to end)")
        self.set_status(f"Running: {label}")
        self.proc = StreamProcessor(
            source, self.make_analyzer(), self.settings,
            on_frame=lambda f: setattr(self, "latest", f),
            on_stats=lambda s: self.q.put(("stats", s)),
            on_log=lambda m: self.q.put(("log", m)),
            on_done=lambda: self.q.put(("done", None)))
        self.proc.start()

    def stop_stream(self):
        if self.proc and self.proc.is_alive():
            self.proc.stop()
            self.proc.join(timeout=3)
        self.proc = None

    def add_zone(self):
        path = filedialog.askopenfilename(title="Pick an image/video showing the camera view",
                                          filetypes=(("Image/Video", "*.jpg *.jpeg *.png *.mp4 *.avi *.mkv *.mov"),))
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            cap = cv2.VideoCapture(path)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                self.log("Could not read a frame from that file.")
                return
        frame = fit_width(frame, 1100)
        h, w = frame.shape[:2]
        x, y, bw, bh = cv2.selectROI("Draw zone, press ENTER to confirm (C to cancel)", frame, showCrosshair=False)
        cv2.destroyAllWindows()
        if bw == 0 or bh == 0:
            return
        name = simpledialog.askstring("Zone name", "Name for this zone (e.g. Desk-1, Server Room):")
        if not name:
            return
        restricted = messagebox.askyesno("Zone type", "Restricted zone?\n\nYes = alert when anyone enters\nNo = work zone (tracks presence / idle time)")
        zones = load_zones(self.settings.zones_file)
        zones.append(Zone(name.strip(), "restricted" if restricted else "work", x / w, y / h, (x + bw) / w, (y + bh) / h))
        save_zones(self.settings.zones_file, zones)
        self.log(f"Zone '{name}' saved ({'restricted' if restricted else 'work'}). Total zones: {len(zones)}")

    def clear_zones(self):
        if messagebox.askyesno("Clear zones", "Delete all saved zones?"):
            save_zones(self.settings.zones_file, [])
            self.log("All zones cleared.")

    def graph(self):
        """Plot people count over time for the most recent session (from outputs/crowd_log.csv)."""
        import csv
        import matplotlib
        matplotlib.use("Agg")  # render to a file; avoids clashing with Tk
        import matplotlib.pyplot as plt

        log_path = Path(self.settings.output_dir) / "crowd_log.csv"
        if not log_path.exists():
            self.log("No data yet. Run a video / camera first, then click 'Crowd Graph'.")
            return
        with open(log_path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            self.log("crowd_log.csv is empty. Run a video / camera first.")
            return
        # keep only the latest session (video time restarts at 0 for each run)
        start = 0
        for i in range(1, len(rows)):
            if float(rows[i]["video_time_s"]) < float(rows[i - 1]["video_time_s"]):
                start = i
        rows = rows[start:]
        t = [float(r["video_time_s"]) for r in rows]
        people = [int(r["people"]) for r in rows]
        weap_t = [float(r["video_time_s"]) for r in rows if int(r["weapons"]) > 0]
        weap_y = [int(r["people"]) for r in rows if int(r["weapons"]) > 0]

        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(t, people, color="#2e86de", linewidth=2, label="People")
        ax.fill_between(t, people, alpha=0.15, color="#2e86de")
        for name, val, col in (("Medium", self.settings.crowd_medium, "#d4a017"),
                               ("High", self.settings.crowd_high, "#e67e22"),
                               ("Critical", self.settings.crowd_critical, "#c0392b")):
            if max(people) >= val * 0.6:  # only draw thresholds that are relevant to the data
                ax.axhline(val, color=col, linestyle="--", linewidth=1, label=f"{name} ({val})")
        if weap_t:
            ax.scatter(weap_t, weap_y, color="red", s=60, zorder=5, label="Weapon detected")
        ax.set_title(f"Crowd count over time (source: {rows[0]['source'][-40:]})")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("People in frame")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
        out = Path(self.settings.output_dir) / "crowd_graph.png"
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)

        img = cv2.imread(str(out))
        self.log(f"Graph saved to {out}   (peak {max(people)}, avg {sum(people) / len(people):.1f})")
        cv2.imshow("Crowd Graph (press any key to close)", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def exit_app(self):
        self.stop_stream()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()