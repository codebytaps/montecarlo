from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, ttk

from PIL import Image, ImageTk

from mc.windrop import enable_file_drop

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RUNNER = HERE / "mc_runner.py"
JOBS = RESULTS / ".jobs"
ACCEPTED = {".py", ".csv", ".parquet", ".pq"}

CHART_ORDER = [
    ("dashboard", "Dashboard"),
    ("fan", "Equity fan"),
    ("drawdown_hist", "Drawdown"),
    ("drawdown_exceedance", "Drawdown odds"),
    ("terminal_equity", "Terminal equity"),
    ("underwater", "Recovery time"),
    ("leverage_sweep", "Leverage sweep"),
]


@dataclass
class UITheme:
    bg: str
    panel: str
    field: str
    text: str
    dim: str
    accent: str
    accent_text: str
    border: str
    ok: str
    bad: str


THEMES = {
    "light": UITheme(bg="#fcfcfb", panel="#f4f3f0", field="#ffffff",
                     text="#0b0b0b", dim="#52514e", accent="#2a78d6",
                     accent_text="#ffffff", border="#dedcd6",
                     ok="#1baf7a", bad="#e34948"),
    "dark": UITheme(bg="#1a1a19", panel="#242422", field="#2e2e2b",
                    text="#ffffff", dim="#c3c2b7", accent="#3987e5",
                    accent_text="#ffffff", border="#3a3a36",
                    ok="#199e70", bad="#e66767"),
}


@dataclass
class RunOutcome:
    name: str
    source: Path
    ok: bool
    summary: dict | None = None
    out_dir: Path | None = None
    error: str = ""


def run_one(path: Path, cfg_kwargs: dict, sweep_levels: list[float] | None,
            themes: list[str], progress) -> RunOutcome:
    out_dir = RESULTS / _safe(path.stem)
    JOBS.mkdir(parents=True, exist_ok=True)
    job_path = JOBS / f"{_safe(path.stem)}.json"
    job_path.write_text(json.dumps({
        "path": str(path), "config": cfg_kwargs, "sweep_levels": sweep_levels,
        "out_dir": str(out_dir), "themes": themes,
    }), encoding="utf-8")

    run_json = out_dir / "run.json"
    if run_json.exists():
        run_json.unlink()

    creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.Popen(
            [sys.executable, str(RUNNER), str(job_path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace", cwd=str(HERE),
            creationflags=creation)
    except OSError as exc:
        return RunOutcome(name=path.stem, source=path, ok=False,
                          error=f"could not start the worker process: {exc}")

    for line in proc.stdout:
        line = line.strip()
        if line.startswith("PROGRESS "):
            progress(line[len("PROGRESS "):])
    stderr = proc.stderr.read()
    proc.wait()

    if run_json.exists():
        report = json.loads(run_json.read_text(encoding="utf-8"))
    else:
        report = {"ok": False, "name": path.stem,
                  "error": stderr.strip() or
                           f"the worker process exited with code {proc.returncode} "
                           f"and wrote no result"}

    if not report.get("ok"):
        return RunOutcome(name=report.get("name") or path.stem, source=path,
                          ok=False, error=report.get("error", "unknown failure"))

    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return RunOutcome(name=report.get("name") or path.stem, source=path, ok=True,
                      summary=summary, out_dir=out_dir)


def _safe(name: str) -> str:
    keep = [c if (c.isalnum() or c in "-_ ") else "_" for c in name]
    return "".join(keep).strip().replace(" ", "_") or "run"


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Monte Carlo Strategy Tester")
        width = max(1100, min(1440, self.winfo_screenwidth() - 160))
        height = max(700, min(880, self.winfo_screenheight() - 220))
        x = (self.winfo_screenwidth() - width) // 2
        y = max(0, (self.winfo_screenheight() - height) // 2 - 40)
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.minsize(1060, 680)

        self.mode = "light"
        self.theme = THEMES[self.mode]
        self.files: list[Path] = []
        self.outcomes: dict[str, RunOutcome] = {}
        self.queue: queue.Queue = queue.Queue()
        self.running = False
        self.has_drop = True
        self._photo = None
        self._resize_job = None
        self._current_chart = "dashboard"

        self.paths_var = tk.IntVar(value=10_000)
        self.horizon_var = tk.StringVar(value="252")
        self.leverage_var = tk.StringVar(value="1.0")
        self.capital_var = tk.StringVar(value="100000")
        self.method_var = tk.StringVar(value="stationary")
        self.ruin_var = tk.StringVar(value="50")
        self.ppy_var = tk.StringVar(value="252")
        self.sweep_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Add a strategy or returns file to begin")

        self._build()
        self._apply_theme()
        self.after(100, self._drain_queue)
        self.after_idle(self._enable_drops)

    def _enable_drops(self) -> None:
        self.has_drop = enable_file_drop(self, self._on_drop)
        self._paint_drop()
        if not self.has_drop:
            self.status_var.set("drag-and-drop unavailable here - use Browse")

    def _build(self) -> None:
        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        self.columnconfigure(0, weight=0, minsize=360)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        self.header = tk.Frame(self)
        self.header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=18, pady=(14, 8))
        self.title_lbl = tk.Label(self.header, text="Monte Carlo",
                                  font=("Segoe UI Semibold", 17), anchor="w")
        self.title_lbl.pack(side="left")
        self.sub_lbl = tk.Label(
            self.header, font=("Segoe UI", 10), anchor="w",
            text="   See how a return history behaves across thousands of possible paths")
        self.sub_lbl.pack(side="left")
        self.theme_btn = tk.Button(self.header, text="Dark", relief="flat", bd=0,
                                   padx=14, pady=4, cursor="hand2",
                                   command=self._toggle_theme)
        self.theme_btn.pack(side="right")

        self._build_left()
        self._build_right()

    def _build_left(self) -> None:
        self.left = tk.Frame(self)
        self.left.grid(row=1, column=0, sticky="nsew", padx=(18, 9), pady=(0, 16))
        self.left.columnconfigure(0, weight=1)
        self.left.rowconfigure(8, weight=1)

        self.drop = tk.Canvas(self.left, height=132, highlightthickness=0, bd=0,
                              cursor="hand2")
        self.drop.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.drop.bind("<Button-1>", lambda _e: self._browse())
        self.drop.bind("<Configure>", lambda _e: self._paint_drop())

        self.files_box = tk.Listbox(self.left, activestyle="none", bd=0,
                                    highlightthickness=1, selectmode="extended",
                                    font=("Consolas", 9), height=6)
        self.files_box.grid(row=1, column=0, sticky="ew")
        self.files_box.bind("<Delete>", lambda _e: self._remove_selected())

        row = tk.Frame(self.left)
        row.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self.remove_btn = tk.Button(row, text="Remove", relief="flat", bd=0,
                                    padx=10, pady=3, cursor="hand2",
                                    command=self._remove_selected)
        self.remove_btn.pack(side="left")
        self.clear_btn = tk.Button(row, text="Clear all", relief="flat", bd=0,
                                   padx=10, pady=3, cursor="hand2",
                                   command=self._clear_files)
        self.clear_btn.pack(side="left", padx=(6, 0))
        self.browse_btn = tk.Button(row, text="Browse...", relief="flat", bd=0,
                                    padx=10, pady=3, cursor="hand2",
                                    command=self._browse)
        self.browse_btn.pack(side="right")

        self.runs_lbl = tk.Label(self.left, text="Simulations", anchor="w",
                                 font=("Segoe UI Semibold", 8))
        self.runs_lbl.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        runs = tk.Frame(self.left)
        runs.grid(row=4, column=0, sticky="ew", pady=(4, 12))
        self.run_btns: dict[int, tk.Button] = {}
        for i, n in enumerate((1_000, 10_000)):
            b = tk.Button(runs, text=f"{n:,}", relief="flat", bd=0, pady=7,
                          cursor="hand2", font=("Segoe UI", 10),
                          command=lambda v=n: self._set_paths(v))
            b.pack(side="left", expand=True, fill="x", padx=(0 if i == 0 else 6, 0))
            self.run_btns[n] = b
        self.custom_entry = tk.Entry(runs, width=8, bd=0, justify="center",
                                     font=("Segoe UI", 10))
        self.custom_entry.pack(side="left", padx=(6, 0), ipady=5)
        self.custom_entry.bind("<Return>", self._set_custom_paths)
        self.custom_entry.bind("<FocusOut>", self._on_custom_blur)
        self.custom_entry.bind("<FocusIn>", self._on_custom_focus)
        self._show_placeholder()

        self.params_lbl = tk.Label(self.left, text="Parameters", anchor="w",
                                   font=("Segoe UI Semibold", 8))
        self.params_lbl.grid(row=5, column=0, sticky="ew")
        grid = tk.Frame(self.left)
        grid.grid(row=6, column=0, sticky="ew", pady=(6, 10))
        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(3, weight=1)

        self.field_labels: list[tk.Label] = []
        self.entries: list[tk.Entry] = []

        def field(r, c, text, var, width=8):
            lbl = tk.Label(grid, text=text, anchor="w", font=("Segoe UI", 9))
            lbl.grid(row=r, column=c, sticky="w", pady=3)
            self.field_labels.append(lbl)
            e = tk.Entry(grid, textvariable=var, width=width, bd=0,
                         font=("Segoe UI", 9), justify="right")
            e.grid(row=r, column=c + 1, sticky="ew", padx=(6, 14), ipady=3)
            self.entries.append(e)

        field(0, 0, "Horizon", self.horizon_var)
        field(0, 2, "Periods/yr", self.ppy_var)
        field(1, 0, "Leverage", self.leverage_var)
        field(1, 2, "Capital", self.capital_var)
        field(2, 0, "Ruin level %", self.ruin_var)

        m_lbl = tk.Label(grid, text="Method", anchor="w", font=("Segoe UI", 9))
        m_lbl.grid(row=2, column=2, sticky="w", pady=3)
        self.field_labels.append(m_lbl)
        self.method_box = ttk.Combobox(
            grid, textvariable=self.method_var, state="readonly", width=10,
            values=["stationary", "iid", "block", "normal", "t"])
        self.method_box.grid(row=2, column=3, sticky="ew", padx=(6, 14))

        self.sweep_chk = tk.Checkbutton(
            self.left, text="Also sweep leverage (0.5x - 5x)", anchor="w",
            variable=self.sweep_var, bd=0, highlightthickness=0,
            font=("Segoe UI", 9), cursor="hand2")
        self.sweep_chk.grid(row=7, column=0, sticky="ew", pady=(0, 4))

        self.spacer = tk.Frame(self.left, height=1)
        self.spacer.grid(row=8, column=0, sticky="nsew")

        self.run_btn = tk.Button(self.left, text="Run", relief="flat", bd=0,
                                 pady=11, cursor="hand2",
                                 font=("Segoe UI Semibold", 11),
                                 command=self._run)
        self.run_btn.grid(row=9, column=0, sticky="ew")

        self.progress = ttk.Progressbar(self.left, mode="determinate")
        self.progress.grid(row=10, column=0, sticky="ew", pady=(8, 4))
        self.status_lbl = tk.Label(self.left, textvariable=self.status_var,
                                   anchor="w", font=("Segoe UI", 9),
                                   wraplength=330, justify="left")
        self.status_lbl.grid(row=11, column=0, sticky="ew")

    def _build_right(self) -> None:
        self.right = tk.Frame(self)
        self.right.grid(row=1, column=1, sticky="nsew", padx=(9, 18), pady=(0, 16))
        self.right.columnconfigure(0, weight=1)
        self.right.rowconfigure(3, weight=1)

        self.tiles = tk.Frame(self.right)
        self.tiles.grid(row=0, column=0, sticky="ew")
        self.tile_widgets: list[tuple[tk.Frame, tk.Label, tk.Label]] = []
        for i in range(5):
            self.tiles.columnconfigure(i, weight=1)
            card = tk.Frame(self.tiles)
            card.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 8, 0))
            value = tk.Label(card, text="-", font=("Segoe UI Semibold", 19), anchor="w")
            value.pack(fill="x", padx=12, pady=(9, 0))
            name = tk.Label(card, text="", font=("Segoe UI", 8), anchor="w")
            name.pack(fill="x", padx=12, pady=(0, 9))
            self.tile_widgets.append((card, value, name))

        self.table = ttk.Treeview(
            self.right, height=4, selectmode="browse",
            columns=("ruin", "cagr", "dd", "loss", "worst"), show="tree headings")
        self.table.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        self.table.heading("#0", text="Strategy")
        self.table.column("#0", width=260, anchor="w")
        for key, text, width in (("ruin", "Ruin chance", 100), ("cagr", "Typical CAGR", 110),
                                 ("dd", "Typical drawdown", 125), ("loss", "Loss chance", 95),
                                 ("worst", "Weak-case CAGR", 115)):
            self.table.heading(key, text=text)
            self.table.column(key, width=width, anchor="e")
        self.table.bind("<<TreeviewSelect>>", lambda _e: self._show_selected())

        self.tabs = tk.Frame(self.right)
        self.tabs.grid(row=2, column=0, sticky="ew", pady=(12, 8))
        self.tab_btns: dict[str, tk.Button] = {}
        for key, text in CHART_ORDER:
            b = tk.Button(self.tabs, text=text, relief="flat", bd=0, padx=12,
                          pady=5, cursor="hand2", font=("Segoe UI", 9),
                          command=lambda k=key: self._select_chart(k))
            b.pack(side="left", padx=(0, 6))
            self.tab_btns[key] = b

        self.canvas_frame = tk.Frame(self.right)
        self.canvas_frame.grid(row=3, column=0, sticky="nsew")
        self.canvas_frame.bind("<Configure>", self._on_resize)
        self.image_lbl = tk.Label(self.canvas_frame, anchor="center",
                                  font=("Segoe UI", 10))
        self.image_lbl.place(relx=0.5, rely=0.5, anchor="center")

    def _apply_theme(self) -> None:
        t = self.theme
        self.configure(bg=t.bg)
        for frame in (self.header, self.left, self.right, self.tiles, self.tabs,
                      self.canvas_frame, self.spacer):
            frame.configure(bg=t.bg)
        for w in self.left.winfo_children() + self.tabs.winfo_children():
            if isinstance(w, tk.Frame):
                w.configure(bg=t.bg)
                for c in w.winfo_children():
                    if isinstance(c, (tk.Frame, tk.Label)):
                        c.configure(bg=t.bg)

        self.title_lbl.configure(bg=t.bg, fg=t.text)
        self.sub_lbl.configure(bg=t.bg, fg=t.dim)
        self.image_lbl.configure(bg=t.bg, fg=t.dim)
        for lbl in (self.runs_lbl, self.params_lbl, *self.field_labels):
            lbl.configure(bg=t.bg, fg=t.dim)
        self.status_lbl.configure(bg=t.bg, fg=t.dim)
        self.sweep_chk.configure(bg=t.bg, fg=t.dim, activebackground=t.bg,
                                 activeforeground=t.text, selectcolor=t.field)

        for b in (self.theme_btn, self.remove_btn, self.clear_btn, self.browse_btn):
            b.configure(bg=t.panel, fg=t.dim, activebackground=t.border,
                        activeforeground=t.text)
        self.theme_btn.configure(text="Light" if self.mode == "dark" else "Dark")
        self.run_btn.configure(bg=t.accent, fg=t.accent_text,
                               activebackground=t.accent, activeforeground=t.accent_text,
                               disabledforeground=t.dim)
        self.files_box.configure(bg=t.field, fg=t.text, highlightbackground=t.border,
                                 highlightcolor=t.border, selectbackground=t.accent,
                                 selectforeground=t.accent_text)
        placeholder = self.custom_entry.get() == self.PLACEHOLDER
        self.custom_entry.configure(bg=t.field, insertbackground=t.text,
                                    fg=t.dim if placeholder else t.text)
        for e in self.entries:
            e.configure(bg=t.field, fg=t.text, insertbackground=t.text)
        for card, value, name in self.tile_widgets:
            card.configure(bg=t.panel)
            value.configure(bg=t.panel, fg=t.text)
            name.configure(bg=t.panel, fg=t.dim)

        self.style.configure("TCombobox", fieldbackground=t.field, background=t.panel,
                             foreground=t.text, arrowcolor=t.dim, bordercolor=t.border)
        self.style.map("TCombobox", fieldbackground=[("readonly", t.field)],
                       foreground=[("readonly", t.text)])
        self.style.configure("Treeview", background=t.field, fieldbackground=t.field,
                             foreground=t.text, bordercolor=t.border, rowheight=24)
        self.style.configure("Treeview.Heading", background=t.panel, foreground=t.dim,
                             relief="flat")
        self.style.map("Treeview", background=[("selected", t.accent)],
                       foreground=[("selected", t.accent_text)])
        self.style.configure("TProgressbar", background=t.accent, troughcolor=t.panel,
                             bordercolor=t.panel, lightcolor=t.accent,
                             darkcolor=t.accent)

        self._paint_run_buttons()
        self._paint_tabs()
        self._paint_drop()

    def _toggle_theme(self) -> None:
        self.mode = "dark" if self.mode == "light" else "light"
        self.theme = THEMES[self.mode]
        self._apply_theme()
        if self.outcomes:
            self._show_selected()

    def _paint_drop(self, hot: bool = False) -> None:
        t = self.theme
        c = self.drop
        c.delete("all")
        c.configure(bg=t.bg)
        w = c.winfo_width() or 340
        h = c.winfo_height() or 132
        c.create_rectangle(2, 2, w - 3, h - 3, dash=(6, 5), width=2,
                           outline=t.accent if hot else t.border,
                           fill=t.panel if hot else t.bg)
        headline = ("Drop a strategy or returns file" if getattr(self, "has_drop", True)
                    else "Click to choose a file")
        c.create_text(w / 2, h / 2 - 12, text=headline, fill=t.text,
                      font=("Segoe UI Semibold", 12))
        detail = ("or click to browse   -   .py, .csv, or .parquet"
                  if getattr(self, "has_drop", True) else ".py, .csv, or .parquet")
        c.create_text(w / 2, h / 2 + 12, text=detail, fill=t.dim,
                      font=("Segoe UI", 9))

    def _paint_run_buttons(self) -> None:
        t = self.theme
        for n, b in self.run_btns.items():
            on = self.paths_var.get() == n
            b.configure(bg=t.accent if on else t.panel,
                        fg=t.accent_text if on else t.dim,
                        activebackground=t.accent if on else t.border,
                        activeforeground=t.accent_text if on else t.text)

    def _paint_tabs(self) -> None:
        t = self.theme
        for key, b in self.tab_btns.items():
            on = key == self._current_chart
            b.configure(bg=t.panel if on else t.bg,
                        fg=t.text if on else t.dim,
                        activebackground=t.panel, activeforeground=t.text)

    def _on_drop(self, paths: list[Path]) -> None:
        self._flash_drop()
        self._add_files(list(paths))

    def _flash_drop(self) -> None:
        self._paint_drop(True)
        self.after(220, lambda: self._paint_drop(False))

    def _browse(self) -> None:
        chosen = filedialog.askopenfilenames(
            title="Choose a strategy or returns file",
            filetypes=[("Strategy or returns", "*.py *.csv *.parquet"),
                       ("Python strategy", "*.py"), ("All files", "*.*")])
        self._add_files([Path(p) for p in chosen])

    def _add_files(self, paths: list[Path]) -> None:
        added, skipped = 0, []
        for p in paths:
            if p.is_dir():
                for child in sorted(p.glob("*.py")):
                    if child not in self.files:
                        self.files.append(child)
                        added += 1
                continue
            if p.suffix.lower() not in ACCEPTED:
                skipped.append(p.name)
                continue
            if p not in self.files:
                self.files.append(p)
                added += 1
        self._refresh_files()
        msg = f"{added} file{'s' if added != 1 else ''} added"
        if skipped:
            msg += f" - ignored {', '.join(skipped[:3])} (use .py, .csv, or .parquet)"
        self.status_var.set(msg)

    def _refresh_files(self) -> None:
        self.files_box.delete(0, "end")
        for p in self.files:
            self.files_box.insert("end", f"  {p.name}")
        self.run_btn.configure(text=f"Run  ({len(self.files)})" if self.files else "Run")

    def _remove_selected(self) -> None:
        for i in sorted(self.files_box.curselection(), reverse=True):
            del self.files[i]
        self._refresh_files()

    def _clear_files(self) -> None:
        self.files.clear()
        self._refresh_files()
        self.status_var.set("cleared")

    def _set_paths(self, n: int) -> None:
        self.paths_var.set(n)
        self._show_placeholder()
        self._paint_run_buttons()

    PLACEHOLDER = "custom"

    def _show_placeholder(self) -> None:
        self.custom_entry.delete(0, "end")
        self.custom_entry.insert(0, self.PLACEHOLDER)
        self.custom_entry.configure(fg=self.theme.dim)

    def _on_custom_focus(self, _event=None) -> None:
        if self.custom_entry.get() == self.PLACEHOLDER:
            self.custom_entry.delete(0, "end")
        self.custom_entry.configure(fg=self.theme.text)

    def _on_custom_blur(self, _event=None) -> None:
        self._set_custom_paths()
        if not self.custom_entry.get().strip():
            self._show_placeholder()

    def _set_custom_paths(self, _event=None) -> None:
        raw = self.custom_entry.get().strip().replace(",", "").replace("_", "")
        if not raw or raw == self.PLACEHOLDER:
            return
        try:
            n = int(float(raw))
        except ValueError:
            self.status_var.set(f"'{raw}' is not a number of simulations")
            return
        if n < 1:
            self.status_var.set("simulations must be at least 1")
            return
        self.paths_var.set(n)
        self._paint_run_buttons()

    def _collect_config(self) -> dict | None:
        def num(var, name, cast=float, lo=None, hi=None):
            raw = var.get().strip().replace(",", "")
            try:
                v = cast(float(raw))
            except ValueError:
                raise ValueError(f"{name}: '{raw}' is not a number")
            if lo is not None and v < lo:
                raise ValueError(f"{name} must be at least {lo}")
            if hi is not None and v > hi:
                raise ValueError(f"{name} must be at most {hi}")
            return v

        try:
            ruin_pct = num(self.ruin_var, "Ruin level %", float, 0, 99.9)
            return dict(
                method=self.method_var.get(),
                n_paths=int(self.paths_var.get()),
                horizon=num(self.horizon_var, "Horizon", int, 1),
                initial_capital=num(self.capital_var, "Capital", float, 0.01),
                leverage=num(self.leverage_var, "Leverage", float, 0.01),
                periods_per_year=num(self.ppy_var, "Periods/yr", float, 0.01),
                ruin_threshold=ruin_pct / 100.0,
                fan_paths=min(5_000, int(self.paths_var.get())),
            )
        except ValueError as exc:
            self.status_var.set(str(exc))
            return None

    def _run(self) -> None:
        if self.running:
            return
        if not self.files:
            self.status_var.set("add a strategy or returns file first")
            return
        self._set_custom_paths()
        cfg_kwargs = self._collect_config()
        if cfg_kwargs is None:
            return

        levels = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0] if self.sweep_var.get() else None
        themes = ["light", "dark"]
        files = list(self.files)

        self.running = True
        self.outcomes.clear()
        self.table.delete(*self.table.get_children())
        self.run_btn.configure(state="disabled", text="Running...")
        self.progress.configure(maximum=len(files), value=0)

        def work():
            for i, path in enumerate(files, start=1):
                self.queue.put(("status", f"[{i}/{len(files)}] {path.name}"))
                outcome = run_one(
                    path, cfg_kwargs, levels, themes,
                    lambda msg: self.queue.put(("status", f"[{i}/{len(files)}] {msg}")))
                self.queue.put(("result", outcome))
                self.queue.put(("progress", i))
            self.queue.put(("done", None))

        threading.Thread(target=work, daemon=True).start()

    def _drain_queue(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "status":
                    self.status_var.set(payload)
                elif kind == "progress":
                    self.progress.configure(value=payload)
                elif kind == "result":
                    self._add_outcome(payload)
                elif kind == "done":
                    self.running = False
                    self.run_btn.configure(state="normal")
                    self._refresh_files()
                    ok = sum(1 for o in self.outcomes.values() if o.ok)
                    bad = len(self.outcomes) - ok
                    msg = f"done - {ok} run{'s' if ok != 1 else ''}"
                    if bad:
                        msg += f", {bad} failed (click the row to see why)"
                    self.status_var.set(msg)
        except queue.Empty:
            pass
        self.after(100, self._drain_queue)

    def _add_outcome(self, out: RunOutcome) -> None:
        key = out.name
        n = 2
        while key in self.outcomes:
            key = f"{out.name} ({n})"
            n += 1
        self.outcomes[key] = out

        if out.ok:
            p = out.summary["probabilities"]
            d = out.summary["distribution"]
            values = (f"{p['ruin']:.2%}", f"{d['cagr']['p50']:+.2%}",
                      f"{d['max_drawdown']['p50']:.1%}", f"{p['loss']:.1%}",
                      f"{d['cagr']['p05']:+.2%}")
        else:
            values = ("failed", "-", "-", "-", "-")
        self.table.insert("", "end", iid=key, text=f" {key}", values=values)
        if len(self.table.get_children()) == 1:
            self.table.selection_set(key)

    def _select_chart(self, key: str) -> None:
        self._current_chart = key
        self._paint_tabs()
        self._show_selected()

    def _show_selected(self) -> None:
        sel = self.table.selection()
        if not sel:
            return
        out = self.outcomes.get(sel[0])
        if out is None:
            return
        if not out.ok:
            self._clear_tiles()
            self._photo = None
            self.image_lbl.configure(image="", text=out.error, justify="left",
                                     wraplength=max(520, self.canvas_frame.winfo_width() - 60))
            return

        self._fill_tiles(out)
        suffix = "_dark" if self.mode == "dark" else ""
        path = out.out_dir / f"{self._current_chart}{suffix}.png"
        if not path.exists():
            hint = ("tick 'Also sweep leverage' and run again"
                    if self._current_chart == "leverage_sweep" else "not generated")
            self._photo = None
            self.image_lbl.configure(image="", text=hint, wraplength=400)
            return
        self._render_image(path)

    def _render_image(self, path: Path) -> None:
        frame_w = max(self.canvas_frame.winfo_width() - 12, 200)
        frame_h = max(self.canvas_frame.winfo_height() - 12, 160)
        try:
            img = Image.open(path)
        except OSError as exc:
            self.image_lbl.configure(image="", text=f"could not open chart: {exc}")
            return
        img.thumbnail((frame_w, frame_h), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self.image_lbl.configure(image=self._photo, text="")

    def _on_resize(self, _event=None) -> None:
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(180, self._show_selected)

    def _fill_tiles(self, out: RunOutcome) -> None:
        p = out.summary["probabilities"]
        d = out.summary["distribution"]
        t = self.theme
        cells = [
            (f"{p['ruin']:.2%}", "chance of ruin",
             t.bad if p["ruin"] >= 0.05 else t.text),
            (f"{d['cagr']['p50']:+.1%}", "median CAGR", t.text),
            (f"{d['max_drawdown']['p50']:.0%}", "median max drawdown", t.text),
            (f"{d['max_drawdown']['p95']:.0%}", "95th pct drawdown", t.text),
            (f"{p['loss']:.0%}", "chance of ending down", t.text),
        ]
        for (card, value, name), (v, n, colour) in zip(self.tile_widgets, cells):
            value.configure(text=v, fg=colour)
            name.configure(text=n)

    def _clear_tiles(self) -> None:
        for _card, value, name in self.tile_widgets:
            value.configure(text="-", fg=self.theme.dim)
            name.configure(text="")


def main() -> int:
    App().mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
