import os
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
from scipy import signal
import matplotlib.pyplot as plt
import smtplib
from email.message import EmailMessage

class CoherenceAnalysisGUI:
    def __init__(self, master):
        self.master = master
        master.title("Batch Coherence Calculator")
        master.geometry("900x650")
        self.recipient_email = None
        self.build_interface()

    def build_interface(self):
        frame_path = ttk.LabelFrame(self.master, text="Folder and Column Selection", padding=10)
        frame_path.pack(fill='x', padx=10, pady=5)

        ttk.Button(frame_path, text="Select Folder", command=self.select_folder).pack(anchor='w')
        self.lbl_folder = ttk.Label(frame_path, text="")
        self.lbl_folder.pack(anchor='w', pady=5)

        self.combo_cols = []
        frame_cols = ttk.Frame(frame_path)
        frame_cols.pack()
        for i in range(2):
            ttk.Label(frame_cols, text=f"Column {i+1}:").grid(row=i, column=0, sticky='e')
            cb = ttk.Combobox(frame_cols, width=30, state="readonly")
            cb.grid(row=i, column=1, padx=5, pady=2)
            self.combo_cols.append(cb)

        frame_sampling = ttk.LabelFrame(self.master, text="Settings", padding=10)
        frame_sampling.pack(fill='x', padx=10, pady=5)

        ttk.Label(frame_sampling, text="Sampling Rate (Hz):").pack(side='left')
        self.entry_fs = ttk.Entry(frame_sampling, width=8)
        self.entry_fs.insert(0, "1000")
        self.entry_fs.pack(side='left', padx=5)

        self.var_window = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_sampling, text="Enable Sliding Window", variable=self.var_window).pack(side='left', padx=5)

        self.var_per_segment = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_sampling, text="Export Per Segment", variable=self.var_per_segment).pack(side='left', padx=5)

        self.var_plot = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_sampling, text="Plot Segment Trends", variable=self.var_plot).pack(side='left', padx=5)

        self.var_email = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_sampling, text="Email Result", variable=self.var_email).pack(side='left', padx=5)

        ttk.Label(frame_sampling, text="Win:").pack(side='left')
        self.entry_window = ttk.Entry(frame_sampling, width=6)
        self.entry_window.insert(0, "1000")
        self.entry_window.pack(side='left')

        ttk.Label(frame_sampling, text="%Ov:").pack(side='left')
        self.entry_overlap = ttk.Entry(frame_sampling, width=5)
        self.entry_overlap.insert(0, "50")
        self.entry_overlap.pack(side='left')

        frame_log = ttk.LabelFrame(self.master, text="Execution Log", padding=10)
        frame_log.pack(fill='both', expand=True, padx=10, pady=5)

        self.log = scrolledtext.ScrolledText(frame_log, height=12)
        self.log.pack(fill='both', expand=True)

        ttk.Button(self.master, text="Start Batch Processing", command=self.start_processing).pack(pady=10)

    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.lbl_folder.config(text=folder)
            files = [f for f in os.listdir(folder) if f.endswith(('.xlsx', '.csv'))]
            if files:
                path = os.path.join(folder, files[0])
                df = pd.read_excel(path) if path.endswith(('xlsx', 'xls')) else pd.read_csv(path)
                cols = df.columns.tolist()
                for cb in self.combo_cols:
                    cb['values'] = [''] + cols
                    cb.set('')
            else:
                messagebox.showwarning("No Files", "No Excel or CSV files found.")

    def log_message(self, msg):
        self.log.insert(tk.END, msg + '\n')
        self.log.yview(tk.END)

    def calculate_coherence(self, X, Y, fs=1000, nperseg=None):
        f, Cxy = signal.coherence(X, Y, fs=fs, nperseg=nperseg)
        return f, Cxy, np.mean(Cxy)

    def start_processing(self):
        folder = self.lbl_folder.cget("text")
        selected_cols = [cb.get() for cb in self.combo_cols if cb.get()]
        if len(selected_cols) != 2:
            messagebox.showerror("Column Error", "Please select exactly 2 columns.")
            return
        try:
            fs = int(self.entry_fs.get())
        except ValueError:
            messagebox.showerror("Error", "Sampling rate must be numeric.")
            return

        use_window = self.var_window.get()
        export_segment = self.var_per_segment.get()
        plot_segment = self.var_plot.get()
        send_email = self.var_email.get()

        if send_email:
            self.recipient_email = simpledialog.askstring("Email", "Enter recipient email:")

        window_size = int(self.entry_window.get()) if use_window else None
        overlap_ratio = float(self.entry_overlap.get()) / 100.0 if use_window else 0

        files = [f for f in os.listdir(folder) if f.endswith(('.xlsx', '.csv'))]
        summary_results = []
        segment_results = {}

        for file in files:
            try:
                path = os.path.join(folder, file)
                df = pd.read_excel(path) if path.endswith(('xlsx', 'xls')) else pd.read_csv(path)
                X = df[selected_cols[0]].dropna().reset_index(drop=True)
                Y = df[selected_cols[1]].dropna().reset_index(drop=True)
                length = min(len(X), len(Y))
                X, Y = X[:length], Y[:length]

                if use_window:
                    step = int(window_size * (1 - overlap_ratio))
                    num_segments = (length - window_size) // step + 1
                    coh_values = []
                    segs = []

                    for i in range(num_segments):
                        s, e = i * step, i * step + window_size
                        if e > length: break
                        f, Cxy, avg = self.calculate_coherence(X[s:e], Y[s:e], fs)
                        coh_values.append(avg)
                        segs.append({"Segment": f"Segment{i+1}", "Coherence": avg, "File": file})

                    summary_results.append({"File": file, "Mean Coherence": np.mean(coh_values)})
                    if export_segment:
                        segment_results[file] = segs
                    if plot_segment:
                        plt.figure()
                        plt.plot(coh_values, marker='o')
                        plt.title(f"{file} - Segment Coherence")
                        plt.xlabel("Segment")
                        plt.ylabel("Coherence")
                        plt.grid()
                        plt.tight_layout()
                        plt.show()

                else:
                    f, Cxy, avg = self.calculate_coherence(X, Y, fs)
                    summary_results.append({"File": file, "Coherence": avg})

                self.log_message(f"Processed {file}")

            except Exception as e:
                self.log_message(f"Error {file}: {e}")

        # Summary output
        summary_path = os.path.join(folder, "Coherence_Summary.xlsx")
        pd.DataFrame(summary_results).to_excel(summary_path, index=False)

        # Segment output
        if export_segment and segment_results:
            seg_path = os.path.join(folder, "Coherence_PerSegment.xlsx")
            with pd.ExcelWriter(seg_path, engine='openpyxl') as writer:
                for file, rows in segment_results.items():
                    df_seg = pd.DataFrame(rows)
                    sheet_name = os.path.splitext(file)[0][:31]
                    df_seg.to_excel(writer, sheet_name=sheet_name, index=False)

        # Email sending
        if send_email and self.recipient_email:
            try:
                files_to_send = [summary_path]
                if export_segment:
                    files_to_send.append(seg_path)
                self.send_email(self.recipient_email, files_to_send)
                self.log_message("✅ Email sent successfully.")
            except Exception as e:
                self.log_message(f"❌ Email failed: {e}")

        messagebox.showinfo("Done", "All files processed.")

    def send_email(self, to_email, attachments):
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        sender_email = "4A930099@stust.edu.tw"
        sender_password = "trwvoyligttxqcjy"

        msg = EmailMessage()
        msg['Subject'] = "Coherence Analysis Report"
        msg['From'] = sender_email
        msg['To'] = to_email
        msg.set_content("Attached are the Coherence analysis results.")

        for path in attachments:
            with open(path, 'rb') as f:
                data = f.read()
                name = os.path.basename(path)
                msg.add_attachment(data, maintype='application', subtype='octet-stream', filename=name)

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)

if __name__ == "__main__":
    root = tk.Tk()
    app = CoherenceAnalysisGUI(root)
    root.mainloop()
