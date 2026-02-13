import os
import threading
import pandas as pd
import numpy as np
import EntropyHub as EH
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk
import sys
import subprocess

class ApproxEntropyApp:
    MAX_COLS = 5

    def __init__(self, master):
        self.master = master
        master.title("Approximate Entropy Calculator")
        master.geometry("850x830")

        container = ttk.Frame(master, padding=10)
        container.pack(fill='both', expand=True)

        # Input/output
        input_frame = ttk.Labelframe(container, text="Input/Output Settings", padding=10)
        input_frame.pack(fill='x', pady=5)
        ttk.Label(input_frame, text="Folder:").grid(row=0, column=0, sticky='w')
        self.entry_folder = ttk.Entry(input_frame, width=60)
        self.entry_folder.grid(row=0, column=1, sticky='ew')
        ttk.Button(input_frame, text="Browse", command=self.browse_folder).grid(row=0, column=2)
        ttk.Button(input_frame, text="Load Columns", command=self.load_columns).grid(row=1, column=1, pady=5)
        ttk.Label(input_frame, text="Output File:").grid(row=2, column=0, sticky='w')
        self.entry_output = ttk.Entry(input_frame, width=60)
        self.entry_output.grid(row=2, column=1, sticky='ew')
        ttk.Button(input_frame, text="Browse", command=self.browse_output).grid(row=2, column=2)
        input_frame.columnconfigure(1, weight=1)

        # Column selection
        column_frame = ttk.Labelframe(container, text="Select Columns (up to 5)", padding=10)
        column_frame.pack(fill='x', pady=5)
        self.combo_cols = []
        for i in range(self.MAX_COLS):
            ttk.Label(column_frame, text=f"Column {i+1}:").grid(row=i, column=0, sticky='e')
            combo = ttk.Combobox(column_frame, state="readonly", width=30)
            combo.grid(row=i, column=1, sticky='w', padx=5, pady=2)
            self.combo_cols.append(combo)
        column_frame.columnconfigure(1, weight=1)

        # Parameters
        param_frame = ttk.Labelframe(container, text="Entropy Parameters", padding=10)
        param_frame.pack(fill='x', pady=5)
        ttk.Label(param_frame, text="Embedding dimension (m):").grid(row=0, column=0, sticky='w')
        self.entry_m = ttk.Entry(param_frame, width=10)
        self.entry_m.insert(0, "2")
        self.entry_m.grid(row=0, column=1, sticky='w', padx=5)

        self.use_window = tk.BooleanVar(value=False)
        ttk.Checkbutton(param_frame, text="Use sliding window", variable=self.use_window, command=self.toggle_window).grid(row=1, column=0, columnspan=2, sticky='w', pady=5)

        self.win_frame = ttk.Frame(param_frame)
        self.win_frame.grid(row=2, column=0, columnspan=3, sticky='w', pady=5)

        ttk.Label(self.win_frame, text="Window size:").grid(row=0, column=0, sticky='w')
        self.entry_win = ttk.Entry(self.win_frame, width=10)
        self.entry_win.insert(0, "1000")
        self.entry_win.grid(row=0, column=1, sticky='w', padx=5)

        ttk.Label(self.win_frame, text="Overlap:").grid(row=1, column=0, sticky='w')
        self.entry_ovl = ttk.Entry(self.win_frame, width=10)
        self.entry_ovl.insert(0, "500")
        self.entry_ovl.grid(row=1, column=1, sticky='w', padx=5)

        ttk.Label(self.win_frame, text="Output style:").grid(row=2, column=0, sticky='w')
        self.output_style = ttk.Combobox(self.win_frame, state="readonly", width=20, values=["Per Segment", "Average Only"])
        self.output_style.set("Per Segment")
        self.output_style.grid(row=2, column=1, sticky='w')

        # Open folder checkbox
        self.open_folder_flag = tk.BooleanVar(value=False)
        ttk.Checkbutton(container, text="Open folder after export", variable=self.open_folder_flag).pack(anchor='w', padx=10, pady=(5, 10))

        # Progress and log
        progress_frame = ttk.Frame(container)
        progress_frame.pack(fill='both', expand=True, pady=5)
        self.progress = ttk.Progressbar(progress_frame, orient="horizontal", mode="determinate")
        self.progress.pack(fill='x', pady=5)
        self.log = scrolledtext.ScrolledText(progress_frame, height=15, wrap='word')
        self.log.pack(fill='both', expand=True)

        # Start button
        ttk.Button(container, text="Start Calculation", command=self.start).pack(pady=10)

        # ⛔ 初始化時關閉滑動視窗參數區塊
        self.toggle_window()

    def toggle_window(self):
        state = tk.NORMAL if self.use_window.get() else tk.DISABLED
        for widget in self.win_frame.winfo_children():
            widget.configure(state=state)

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
            messagebox.showerror("Invalid input", "Embedding dimension must be numeric.")
            return
        cols = [c.get() for c in self.combo_cols if c.get()]
        if not os.path.isdir(folder) or not cols or not output:
            messagebox.showerror("Missing info", "Ensure folder, columns, and output are set.")
            return
        win_flag = self.use_window.get()
        win = int(self.entry_win.get()) if win_flag else None
        ovl = int(self.entry_ovl.get()) if win_flag else None
        out_style = self.output_style.get() if win_flag else None
        open_folder = self.open_folder_flag.get()
        threading.Thread(target=self.process_files, args=(folder, output, m, cols, win_flag, win, ovl, out_style, open_folder), daemon=True).start()

    def process_files(self, folder, output, m, cols, win_flag, win, ovl, out_style, open_folder):
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.xlsx', '.csv'))]
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

                    if win_flag:
                        seg_results = []
                        for start in range(0, len(data) - win + 1, win - ovl):
                            segment = data[start:start + win]
                            if len(segment) < m + 1: continue
                            r = 0.2 * np.std(segment)
                            val = EH.ApEn(segment, m, r=r)[0][-1]
                            seg_results.append(val)
                        if out_style == "Average Only":
                            row[f"{col}_ApEn_avg"] = np.nanmean(seg_results)
                        else:
                            for i, val in enumerate(seg_results):
                                row[f"{col}_seg{i+1}"] = val
                        logs.append(f"{col} ({len(seg_results)} segments)")
                    else:
                        r = 0.2 * np.std(data)
                        val = EH.ApEn(data, m, r=r)[0][-1]
                        row[f"{col} ApEn"] = val
                        logs.append(f"{col}={val:.4f}")

                results.append(row)
                self.log_message(f"{basename}: " + ", ".join(logs))
            except Exception as e:
                self.log_message(f"Error {basename}: {e}")
            self.progress['value'] += 1

        if results:
            pd.DataFrame(results).to_excel(output, index=False)
            self.log_message(f"Results saved to {output}")
            messagebox.showinfo("Completed", "Calculation finished.")

            if open_folder:
                try:
                    folder_path = os.path.dirname(output)
                    if os.name == 'nt':
                        os.startfile(folder_path)
                    elif sys.platform == 'darwin':
                        subprocess.Popen(['open', folder_path])
                    else:
                        subprocess.Popen(['xdg-open', folder_path])
                except Exception as e:
                    self.log_message(f"Could not open folder: {e}")
        else:
            messagebox.showwarning("No Data", "No valid files processed.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ApproxEntropyApp(root)
    root.mainloop()
