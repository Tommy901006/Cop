import os
import pandas as pd
import numpy as np
import threading
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox, scrolledtext
from EntropyHub import MSEn, MSobject

class MSEnGUI:
    def __init__(self, master):
        self.master = master
        master.title("Batch MSEn Analysis Tool")
        master.geometry("1020x800")
        self.combo_cols = []
        self.build_interface()

    def build_interface(self):
        theme_frame = ttk.Frame(self.master)
        theme_frame.pack(pady=(10, 0))
        ttk.Label(theme_frame, text="Theme:").pack(side="left")
        self.theme_combo = ttk.Combobox(theme_frame, values=["flatly", "darkly"], state="readonly", width=20)
        self.theme_combo.set("flatly")
        self.theme_combo.pack(side="left", padx=5)
        self.theme_combo.bind("<<ComboboxSelected>>", self.change_theme)

        path_frame = ttk.LabelFrame(self.master, text="Folder and Column Selection", padding=10)
        path_frame.pack(fill='x', padx=10, pady=5)
        ttk.Button(path_frame, text="Select Folder", command=self.select_folder).pack(anchor='w')
        self.lbl_folder = ttk.Label(path_frame, text="", bootstyle="secondary")
        self.lbl_folder.pack(anchor='w', pady=5)

        col_frame = ttk.Frame(path_frame)
        col_frame.pack()
        for i in range(5):
            ttk.Label(col_frame, text=f"Column {i+1}:").grid(row=i, column=0, sticky='e')
            cb = ttk.Combobox(col_frame, width=30, state="readonly")
            cb.grid(row=i, column=1, padx=5, pady=2)
            self.combo_cols.append(cb)

        param_frame = ttk.LabelFrame(self.master, text="Analysis Parameters", padding=10)
        param_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(param_frame, text="Embedding Dimension m:").grid(row=0, column=0, sticky='e')
        self.entry_m = ttk.Entry(param_frame, width=5)
        self.entry_m.insert(0, "2")
        self.entry_m.grid(row=0, column=1)

        ttk.Label(param_frame, text="Scales:").grid(row=0, column=2, sticky='e')
        self.entry_scales = ttk.Entry(param_frame, width=5)
        self.entry_scales.insert(0, "5")
        self.entry_scales.grid(row=0, column=3)

        self.use_window = ttk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="Use Sliding Window", variable=self.use_window, command=self.toggle_window).grid(row=0, column=4, padx=15)

        ttk.Label(param_frame, text="Window Size:").grid(row=1, column=0, sticky='e')
        self.entry_win = ttk.Entry(param_frame, width=6)
        self.entry_win.insert(0, "1000")
        self.entry_win.grid(row=1, column=1)

        ttk.Label(param_frame, text="Overlap:").grid(row=1, column=2, sticky='e')
        self.entry_ovl = ttk.Entry(param_frame, width=6)
        self.entry_ovl.insert(0, "500")
        self.entry_ovl.grid(row=1, column=3)

        ttk.Label(param_frame, text="Output Style:").grid(row=1, column=4, sticky='e')
        self.output_style = ttk.Combobox(param_frame, state="readonly", width=20, values=["Average Only", "Per Segment"])
        self.output_style.set("Average Only")
        self.output_style.grid(row=1, column=5, padx=10)

        log_frame = ttk.LabelFrame(self.master, text="Execution Log", padding=10)
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        self.progress = ttk.Progressbar(log_frame, bootstyle="info-striped", mode="determinate")
        self.progress.pack(fill='x', padx=5, pady=5)
        self.log = scrolledtext.ScrolledText(log_frame, height=15)
        self.log.pack(fill='both', expand=True)

        ttk.Button(self.master, text="Start Batch Analysis", bootstyle=PRIMARY, command=self.start_thread).pack(pady=10)
        self.toggle_window()

    def change_theme(self, event=None):
        self.master.style.theme_use(self.theme_combo.get())

    def toggle_window(self):
        state = "normal" if self.use_window.get() else "disabled"
        self.entry_win.configure(state=state)
        self.entry_ovl.configure(state=state)
        self.output_style.configure(state=state)

    def safe_read_file(self, path):
        if path.endswith(".xlsx"):
            return pd.read_excel(path, engine="openpyxl")
        elif path.endswith(".csv"):
            return pd.read_csv(path)
        else:
            raise ValueError("Unsupported file type")

    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.lbl_folder.config(text=folder)
            files = [f for f in os.listdir(folder) if f.endswith(('.xlsx', '.csv'))]
            if files:
                df = self.safe_read_file(os.path.join(folder, files[0]))
                self.file_columns = df.columns.tolist()
                for cb in self.combo_cols:
                    cb['values'] = [''] + self.file_columns
                    cb.set('')

    def log_message(self, msg):
        self.log.insert("end", msg + '\n')
        self.log.yview("end")

    def start_thread(self):
        threading.Thread(target=self.start_analysis, daemon=True).start()

    def start_analysis(self):
        folder = self.lbl_folder.cget("text")
        m = int(self.entry_m.get())
        scales = int(self.entry_scales.get())
        win = int(self.entry_win.get())
        ovl = int(self.entry_ovl.get())
        r = 0.2
        out_style = self.output_style.get()
        selected_cols = [cb.get() for cb in self.combo_cols if cb.get()]
        files = [f for f in os.listdir(folder) if f.endswith(('.xlsx', '.csv'))]
        self.progress['maximum'] = len(files)

        results_per_col = {col: [] for col in selected_cols}

        for idx, file in enumerate(files):
            path = os.path.join(folder, file)
            df = self.safe_read_file(path)

            for col in selected_cols:
                if col not in df.columns:
                    continue
                data = df[col].dropna().values

                if out_style == "Average Only":
                    segs = []
                    for start in range(0, len(data) - win + 1, win - ovl):
                        seg = data[start:start + win]
                        if len(seg) < m + 1: continue
                        Mobj = MSobject('SampEn', m=m, r=r * np.std(seg))
                        MSx, _ = MSEn(seg, Mobj, Scales=scales, Plotx=False)
                        segs.append(MSx)
                    if segs:
                        avg = np.mean(segs, axis=0)
                        row = {"File": file}
                        for i, v in enumerate(avg):
                            row[f"MSE_Scale{i+1}"] = v
                        row["MSE_avg"] = np.mean(avg)
                        results_per_col[col].append(row)

                else:  # Per Segment – Each row: File, MSE_Scale1_Segment1, ..., MSE_ScaleN_SegmentM
                    if file not in [r.get("File") for r in results_per_col[col]]:
                        row = {"File": file}
                    else:
                        row = next(r for r in results_per_col[col] if r["File"] == file)

                    seg_id = 1
                    for start in range(0, len(data) - win + 1, win - ovl):
                        seg = data[start:start + win]
                        if len(seg) < m + 1: continue
                        Mobj = MSobject('SampEn', m=m, r=r * np.std(seg))
                        MSx, _ = MSEn(seg, Mobj, Scales=scales, Plotx=False)
                        for i, v in enumerate(MSx):
                            row[f"MSE_Scale{i+1}_Segment{seg_id}"] = v
                        seg_id += 1
                    if row not in results_per_col[col]:
                        results_per_col[col].append(row)

            self.progress['value'] = idx + 1
            self.log_message(f"✔ {file}")

        out_path = os.path.join(folder, f"MSEn_{out_style.replace(' ', '')}_Sheets.xlsx")
        with pd.ExcelWriter(out_path) as writer:
            for col, data in results_per_col.items():
                if data:
                    df_col = pd.DataFrame(data)
                    df_col.to_excel(writer, sheet_name=col[:31], index=False)

        self.log_message(f"✅ Saved to {out_path}")
        messagebox.showinfo("Done", f"{out_style} analysis completed.")

# Run GUI
app = ttk.Window(themename="flatly")
MSEnGUI(app)
app.mainloop()
