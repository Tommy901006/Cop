import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext, simpledialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import smtplib
from email.message import EmailMessage

class PearsonApp:
    def __init__(self, master):
        self.master = master
        master.title("Pearson Correlation Calculator")
        master.geometry("800x800")

        # Email 資訊
        self.recipient_email = None

        container = ttk.Frame(master, padding=10)
        container.pack(fill='both', expand=True)

        input_frame = ttk.Labelframe(container, text="Input/Output Settings", padding=10)
        input_frame.pack(fill='x', pady=5)

        ttk.Label(input_frame, text="Folder:").grid(row=0, column=0, sticky='w')
        self.entry_folder = ttk.Entry(input_frame, width=60)
        self.entry_folder.grid(row=0, column=1, sticky='ew')
        ttk.Button(input_frame, text="Browse", command=self.browse_folder).grid(row=0, column=2)

        ttk.Button(input_frame, text="Load Columns", command=self.load_columns).grid(row=1, column=1, pady=5)
        input_frame.columnconfigure(1, weight=1)

        column_frame = ttk.Labelframe(container, text="Select 2 Columns", padding=10)
        column_frame.pack(fill='x', pady=5)

        ttk.Label(column_frame, text="Column X:").grid(row=0, column=0, sticky='e')
        self.combo_col_x = ttk.Combobox(column_frame, state="readonly", width=30)
        self.combo_col_x.grid(row=0, column=1, sticky='w', padx=5, pady=2)

        ttk.Label(column_frame, text="Column Y:").grid(row=1, column=0, sticky='e')
        self.combo_col_y = ttk.Combobox(column_frame, state="readonly", width=30)
        self.combo_col_y.grid(row=1, column=1, sticky='w', padx=5, pady=2)

        window_frame = ttk.Labelframe(container, text="Sliding Window Options", padding=10)
        window_frame.pack(fill='x', pady=5)

        self.use_window_var = tk.BooleanVar()
        ttk.Checkbutton(window_frame, text="Enable Sliding Window", variable=self.use_window_var,
                        command=self.toggle_window_inputs).grid(row=0, column=0, sticky='w')

        ttk.Label(window_frame, text="Window Size:").grid(row=1, column=0, sticky='e')
        self.entry_window_size = ttk.Entry(window_frame, width=10)
        self.entry_window_size.grid(row=1, column=1, sticky='w')
        self.entry_window_size.insert(0, "100")

        ttk.Label(window_frame, text="Overlap (%):").grid(row=2, column=0, sticky='e')
        self.entry_overlap = ttk.Entry(window_frame, width=10)
        self.entry_overlap.grid(row=2, column=1, sticky='w')
        self.entry_overlap.insert(0, "50")

        self.use_per_segment_var = tk.BooleanVar()
        self.use_plot_var = tk.BooleanVar()
        self.use_email_var = tk.BooleanVar()

        ttk.Checkbutton(window_frame, text="Export Per Segment Results", variable=self.use_per_segment_var).grid(row=3, column=0, sticky='w')
        ttk.Checkbutton(window_frame, text="Plot Correlation Trends", variable=self.use_plot_var).grid(row=4, column=0, sticky='w')
        ttk.Checkbutton(window_frame, text="Send Results via Email", variable=self.use_email_var).grid(row=5, column=0, sticky='w')

        self.toggle_window_inputs()

        progress_frame = ttk.Frame(container, padding=0)
        progress_frame.pack(fill='both', expand=True, pady=5)

        self.progress = ttk.Progressbar(progress_frame, orient="horizontal", mode="determinate")
        self.progress.pack(fill='x', pady=5)

        self.log = scrolledtext.ScrolledText(progress_frame, height=15, wrap='word')
        self.log.pack(fill='both', expand=True)

        ttk.Button(container, text="Start Calculation", command=self.start).pack(pady=10)

    def toggle_window_inputs(self):
        state = 'normal' if self.use_window_var.get() else 'disabled'
        self.entry_window_size.configure(state=state)
        self.entry_overlap.configure(state=state)

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.entry_folder.delete(0, tk.END)
            self.entry_folder.insert(0, folder)
            self.combo_col_x['values'] = []
            self.combo_col_y['values'] = []
            self.combo_col_x.set('')
            self.combo_col_y.set('')

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
            cols = [''] + list(df.columns.str.strip())
            self.combo_col_x['values'] = cols
            self.combo_col_y['values'] = cols
            self.combo_col_x.set('')
            self.combo_col_y.set('')
            messagebox.showinfo("Columns Loaded", f"Loaded columns from {files[0]}")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def log_message(self, msg):
        self.log.insert(tk.END, msg + "\n")
        self.log.yview(tk.END)

    def start(self):
        folder = self.entry_folder.get()
        col_x = self.combo_col_x.get()
        col_y = self.combo_col_y.get()
        if not os.path.isdir(folder) or not col_x or not col_y:
            messagebox.showerror("Missing info", "Ensure folder and two columns are selected.")
            return
        if self.use_email_var.get():
            self.recipient_email = simpledialog.askstring("Email", "Enter recipient email:")
        threading.Thread(target=self.process_files, args=(folder, col_x, col_y), daemon=True).start()
    def pearson_correlation(self, X, Y):
        if len(X) == len(Y):
            Sum_xy = sum((X - X.mean()) * (Y - Y.mean()))
            Sum_x_squared = sum((X - X.mean()) ** 2)
            Sum_y_squared = sum((Y - Y.mean()) ** 2)
            return Sum_xy / np.sqrt(Sum_x_squared * Sum_y_squared)
        else:
            raise ValueError("X 與 Y 的長度不相等")

    def process_files(self, folder, col_x, col_y):
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.xlsx', '.csv'))]
        self.progress['maximum'] = len(files)
        self.progress['value'] = 0

        use_window = self.use_window_var.get()
        per_segment = self.use_per_segment_var.get()
        plot_result = self.use_plot_var.get()
        send_email = self.use_email_var.get()

        avg_results = []
        segment_results = []

        if use_window:
            try:
                window_size = int(self.entry_window_size.get())
                overlap = float(self.entry_overlap.get()) / 100.0
                if not (0 <= overlap < 1):
                    raise ValueError("Overlap 必須介於 0 到 100% 之間")
            except Exception as e:
                self.log_message(f"參數錯誤: {e}")
                return

        for file in files:
            basename = os.path.basename(file)
            try:
                df = pd.read_excel(file) if file.endswith(('.xls', '.xlsx')) else pd.read_csv(file)
                df.columns = df.columns.str.strip().str.upper()
                col_x_upper = col_x.strip().upper()
                col_y_upper = col_y.strip().upper()

                if col_x_upper not in df.columns or col_y_upper not in df.columns:
                    self.log_message(f"{basename}: missing selected columns.")
                    continue

                X = df[col_x_upper].dropna().reset_index(drop=True)
                Y = df[col_y_upper].dropna().reset_index(drop=True)
                min_len = min(len(X), len(Y))
                if min_len < 1:
                    self.log_message(f"{basename}: not enough data.")
                    continue

                X = X[:min_len]
                Y = Y[:min_len]

                if use_window:
                    step = int(window_size * (1 - overlap))
                    num_segments = (min_len - window_size) // step + 1
                    segment_corrs = []

                    for i in range(num_segments):
                        start = i * step
                        end = start + window_size
                        if end > min_len:
                            break
                        segment_X = X[start:end]
                        segment_Y = Y[start:end]
                        try:
                            corr = self.pearson_correlation(segment_X, segment_Y)
                            segment_corrs.append(corr)
                            if per_segment:
                                segment_results.append({
                                    "File": basename,
                                    "Segment": f"Segment{i+1}",
                                    f"Pearson({col_x},{col_y})": corr
                                })
                        except Exception as e:
                            self.log_message(f"{basename} 段落 {i+1} 發生錯誤: {e}")

                    if segment_corrs:
                        avg_corr = np.mean(segment_corrs)
                        avg_results.append({
                            "檔名": basename,
                            f"Avg Pearson({col_x},{col_y})": avg_corr
                        })
                        self.log_message(f"{basename}: {num_segments} 段落，平均相關係數={avg_corr:.4f}")

                        if plot_result:
                            plt.figure()
                            plt.plot(segment_corrs, marker='o')
                            plt.title(f"{basename} - Pearson Correlation (Segments)")
                            plt.xlabel("Segment Index")
                            plt.ylabel("Correlation")
                            plt.grid(True)
                            plt.show()

                    else:
                        self.log_message(f"{basename}: 無有效段落")
                else:
                    corr = self.pearson_correlation(X, Y)
                    avg_results.append({
                        "檔名": basename,
                        f"Pearson({col_x},{col_y})": corr
                    })
                    self.log_message(f"Processed: {basename}")

            except Exception as e:
                self.log_message(f"Error {basename}: {e}")
            self.progress['value'] += 1

        # 寫入 Excel
        if avg_results:
            avg_df = pd.DataFrame(avg_results)
            avg_path = os.path.join(folder, "Pearson_Avg.xlsx")
            avg_df.to_excel(avg_path, index=False)

        if segment_results:
            seg_path = os.path.join(folder, "Pearson_PerSegment.xlsx")
            with pd.ExcelWriter(seg_path, engine='openpyxl') as writer:
                files_grouped = {}
                for row in segment_results:
                    file = row["File"]
                    if file not in files_grouped:
                        files_grouped[file] = []
                    files_grouped[file].append(row)

                for file, rows in files_grouped.items():
                    sheet_df = pd.DataFrame(rows)
                    sheet_name = os.path.splitext(file)[0][:31]  # Excel 限 31 字元
                    sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)


        if avg_results or segment_results:
            self.log_message("分析完成，結果已儲存。")
            messagebox.showinfo("Done", "Analysis complete. Results saved.")

            if send_email and self.recipient_email:
                try:
                    files_to_send = [avg_path]
                    if per_segment:
                        files_to_send.append(seg_path)
                    self.send_email(self.recipient_email, files_to_send)
                    self.log_message("已成功寄出 Email 報告")
                except Exception as e:
                    self.log_message(f"Email 發送失敗: {e}")
        else:
            messagebox.showwarning("No Data", "No valid files processed.")

    def send_email(self, to_email, attachments):
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        sender_email = "4A930099@stust.edu.tw"
        sender_password = "trwvoyligttxqcjy"

        msg = EmailMessage()
        msg['Subject'] = "Pearson Analysis Report"
        msg['From'] = sender_email
        msg['To'] = to_email
        msg.set_content("分析完成，請查收附檔 Excel 結果。")

        for filepath in attachments:
            with open(filepath, 'rb') as f:
                file_data = f.read()
                file_name = os.path.basename(filepath)
                msg.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)

if __name__ == "__main__":
    root = tk.Tk()
    app = PearsonApp(root)
    root.mainloop()
