import os
import threading
import pandas as pd
import numpy as np
import nolds
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk

class EntropyApp:
    MAX_COLS = 5

    def __init__(self, master):
        self.master = master
        master.title("Sample Entropy Calculator")
        master.geometry("850x930")
        style = ttk.Style(master)
        style.theme_use('clam')

        container = ttk.Frame(master, padding=10)
        container.pack(fill='both', expand=True)

        # Folder frame
        folder_frame = ttk.Labelframe(container, text="Input Files", padding=10)
        folder_frame.pack(fill='x', pady=5)
        ttk.Label(folder_frame, text="Folder:").grid(row=0, column=0, sticky='w')
        self.entry_folder = ttk.Entry(folder_frame, width=60)
        self.entry_folder.grid(row=0, column=1, sticky='ew')
        ttk.Button(folder_frame, text="Browse", command=self.browse_folder).grid(row=0, column=2)
        folder_frame.columnconfigure(1, weight=1)

        ttk.Button(container, text="Load Columns", command=self.load_columns).pack(pady=5)

        # Column selection
        col_frame = ttk.Labelframe(container, text="Select Columns (up to 5)", padding=10)
        col_frame.pack(fill='x', pady=5)
        self.combo_cols = []
        for i in range(self.MAX_COLS):
            ttk.Label(col_frame, text=f"Column {i+1}:").grid(row=i, column=0, sticky='e')
            combo = ttk.Combobox(col_frame, state="readonly", width=30)
            combo.grid(row=i, column=1, sticky='w', padx=5, pady=2)
            self.combo_cols.append(combo)
        col_frame.columnconfigure(1, weight=1)

        # Parameters
        param_frame = ttk.Labelframe(container, text="Parameters", padding=10)
        param_frame.pack(fill='x', pady=5)
        ttk.Label(param_frame, text="Embedding dimension (m):").grid(row=0, column=0, sticky='w')
        self.entry_m = ttk.Entry(param_frame, width=10)
        self.entry_m.insert(0, "1")
        self.entry_m.grid(row=0, column=1, sticky='w', padx=5)

        self.use_window = tk.BooleanVar(value=False)
        ttk.Checkbutton(param_frame, text="Use sliding window", variable=self.use_window, command=self.toggle_window_options).grid(row=1, column=0, columnspan=3, sticky='w', pady=5)

        self.win_enabled = tk.Frame(param_frame)
        self.win_enabled.grid(row=2, column=0, columnspan=3, sticky='w')

        ttk.Label(self.win_enabled, text="Window size:").grid(row=0, column=0, sticky='w')
        self.entry_win = ttk.Entry(self.win_enabled, width=10)
        self.entry_win.insert(0, "1000")
        self.entry_win.grid(row=0, column=1, sticky='w', padx=5)

        ttk.Label(self.win_enabled, text="Overlap:").grid(row=1, column=0, sticky='w')
        self.entry_ovl = ttk.Entry(self.win_enabled, width=10)
        self.entry_ovl.insert(0, "500")
        self.entry_ovl.grid(row=1, column=1, sticky='w', padx=5)

        ttk.Label(self.win_enabled, text="Output style:").grid(row=2, column=0, sticky='w')
        self.output_style = ttk.Combobox(self.win_enabled, state="readonly", width=20, values=["Per Segment", "Average Only"])
        self.output_style.set("Per Segment")
        self.output_style.grid(row=2, column=1, sticky='w')

        # Output path
        ttk.Label(param_frame, text="Output File:").grid(row=6, column=0, sticky='w', pady=5)
        self.entry_output = ttk.Entry(param_frame, width=60)
        self.entry_output.grid(row=6, column=1, sticky='ew')
        ttk.Button(param_frame, text="Browse", command=self.browse_output).grid(row=6, column=2)
        param_frame.columnconfigure(1, weight=1)

        # Log & progress
        progress_frame = ttk.Frame(container)
        progress_frame.pack(fill='both', expand=True, pady=5)
        self.progress = ttk.Progressbar(progress_frame, orient="horizontal", mode="determinate")
        self.progress.pack(fill='x', pady=5)
        self.log = scrolledtext.ScrolledText(progress_frame, height=15, wrap='word')
        self.log.pack(fill='both', expand=True)

        # Start button
        ttk.Button(container, text="Start Calculation", command=self.start).pack(pady=10)

        # 一開始先禁用 window 設定欄位
        self.toggle_window_options()

    def toggle_window_options(self):
        state = tk.NORMAL if self.use_window.get() else tk.DISABLED
        for child in self.win_enabled.winfo_children():
            child.configure(state=state)

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.entry_folder.delete(0, tk.END)
            self.entry_folder.insert(0, folder)
            for combo in self.combo_cols:
                combo['values'] = []
                combo.set('')

    def load_columns(self):
        folder = self.entry_folder.get()
        if not os.path.isdir(folder):
            messagebox.showerror("Invalid folder", "Please select a valid folder first.")
            return
        files = [f for f in os.listdir(folder) if f.lower().endswith((".xls", ".xlsx", ".csv"))]
        if not files:
            messagebox.showwarning("No files found", "No Excel/CSV files in the folder.")
            return
        try:
            path = os.path.join(folder, files[0])
            df = pd.read_excel(path) if path.endswith(('.xls', '.xlsx')) else pd.read_csv(path)
            cols = [''] + list(df.columns)
            for combo in self.combo_cols:
                combo['values'] = cols
                combo.set('')
            messagebox.showinfo("Columns Loaded", f"Loaded columns from {files[0]}")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def browse_output(self):
        file = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if file:
            self.entry_output.delete(0, tk.END)
            self.entry_output.insert(0, file)

    def log_message(self, msg):
        self.log.insert(tk.END, msg + "\n")
        self.log.yview(tk.END)

    def start(self):
        folder = self.entry_folder.get()
        output = self.entry_output.get()
        try:
            m = int(self.entry_m.get())
        except ValueError:
            messagebox.showerror("Invalid m", "Embedding dimension must be integer.")
            return
        cols = [c.get() for c in self.combo_cols if c.get()]
        if not os.path.isdir(folder) or not cols or not output:
            messagebox.showerror("Missing info", "Ensure folder, columns, and output are set.")
            return
        use_window = self.use_window.get()
        win_size = int(self.entry_win.get()) if use_window else None
        overlap = int(self.entry_ovl.get()) if use_window else None
        out_style = self.output_style.get() if use_window else None
        threading.Thread(target=self.process_files, args=(folder, output, m, cols, use_window, win_size, overlap, out_style), daemon=True).start()

    def process_files(self, folder, output, m, cols, use_window, win_size, overlap, out_style):
        files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.lower().endswith((".xls", ".xlsx", ".csv"))]
        self.progress['maximum'] = len(files)
        self.progress['value'] = 0
        results = []

        for file in files:
            basename = os.path.basename(file)
            try:
                df = pd.read_excel(file) if file.endswith(('.xls', '.xlsx')) else pd.read_csv(file)
                row = {'Filename': basename}
                logs = []
                for col in cols:
                    if col not in df.columns:
                        logs.append(f"{col} skipped")
                        continue

                    data = df[col].dropna().values
                    if use_window:
                        s_list = []
                        for start in range(0, len(data) - win_size + 1, win_size - overlap):
                            seg = data[start:start+win_size]
                            if len(seg) < m + 1: continue
                            try:
                                s = nolds.sampen(seg, emb_dim=m)
                                s_list.append(s)
                            except:
                                s_list.append(np.nan)

                        if out_style == "Average Only":
                            row[f"{col} SampEn_avg"] = np.nanmean(s_list)
                        else:
                            for i, val in enumerate(s_list):
                                row[f"{col}_seg{i+1}"] = val
                        logs.append(f"{col} ({len(s_list)} segments)")
                    else:
                        s = nolds.sampen(data, emb_dim=m)
                        row[f"{col} SampEn"] = s
                        logs.append(f"{col}={s:.4f}")
                results.append(row)
                self.log_message(f"{basename}: " + ", ".join(logs))
            except Exception as e:
                self.log_message(f"Error {basename}: {e}")
            self.progress['value'] += 1

        if results:
            pd.DataFrame(results).to_excel(output, index=False)
            self.log_message(f"Saved to {output}")
            messagebox.showinfo("Completed", "Calculation finished.")
        else:
            messagebox.showwarning("No Data", "No valid files processed.")

if __name__ == "__main__":
    root = tk.Tk()
    app = EntropyApp(root)
    root.mainloop()
