import os
import glob
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd

class ConverterApp:
    def __init__(self, master):
        self.master = master
        master.title("批次 TXT 轉檔工具")
        master.geometry("600x400")

        # 選擇資料夾
        self.lbl_folder = tk.Label(master, text="請選擇 TXT 檔案所在資料夾：")
        self.lbl_folder.pack(pady=5, anchor='w')
        self.frm_folder = tk.Frame(master)
        self.frm_folder.pack(fill='x', padx=10)
        self.entry_folder = tk.Entry(self.frm_folder)
        self.entry_folder.pack(side='left', fill='x', expand=True)
        self.btn_browse = tk.Button(self.frm_folder, text="瀏覽…", command=self.browse_folder)
        self.btn_browse.pack(side='right', padx=5)

        # 分隔符設定
        self.lbl_sep = tk.Label(master, text="請輸入分隔符（如 \\t、空格、, 等）：")
        self.lbl_sep.pack(pady=5, anchor='w', padx=10)
        self.entry_sep = tk.Entry(master)
        self.entry_sep.insert(0, "\\t")
        self.entry_sep.pack(fill='x', padx=10)

        # 輸出格式選擇
        self.lbl_format = tk.Label(master, text="請選擇輸出格式：")
        self.lbl_format.pack(pady=5, anchor='w', padx=10)
        self.var_format = tk.StringVar(value="csv")
        frm_fmt = tk.Frame(master)
        frm_fmt.pack(anchor='w', padx=20)
        tk.Radiobutton(frm_fmt, text="CSV", variable=self.var_format, value="csv").pack(side='left', padx=5)
        tk.Radiobutton(frm_fmt, text="XLSX", variable=self.var_format, value="xlsx").pack(side='left', padx=5)
        tk.Radiobutton(frm_fmt, text="CSV + XLSX", variable=self.var_format, value="both").pack(side='left', padx=5)

        # 開始轉檔按鈕
        self.btn_start = tk.Button(master, text="開始轉檔", command=self.start_conversion)
        self.btn_start.pack(pady=10)

        # 日誌輸出區
        self.txt_log = scrolledtext.ScrolledText(master, state='disabled', height=10)
        self.txt_log.pack(fill='both', expand=True, padx=10, pady=5)

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.entry_folder.delete(0, tk.END)
            self.entry_folder.insert(0, folder)

    def log(self, msg):
        self.txt_log.config(state='normal')
        self.txt_log.insert(tk.END, msg + "\n")
        self.txt_log.see(tk.END)
        self.txt_log.config(state='disabled')

    def start_conversion(self):
        folder = self.entry_folder.get().strip()
        sep = self.entry_sep.get().encode('utf-8').decode('unicode_escape')  # 解析轉義字元
        fmt = self.var_format.get()

        if not folder or not os.path.isdir(folder):
            messagebox.showerror("錯誤", "請選擇有效的資料夾！")
            return

        # 禁用按鈕，避免重複點擊
        self.btn_start.config(state='disabled')
        threading.Thread(target=self.convert_files, args=(folder, sep, fmt), daemon=True).start()

    def convert_files(self, folder, sep, fmt):
        txt_files = glob.glob(os.path.join(folder, '*.txt'))
        if not txt_files:
            self.log("在指定資料夾中未找到任何 .txt 檔案。")
        else:
            for txt in txt_files:
                try:
                    df = pd.read_csv(txt, sep=sep, header=None, encoding='utf-8')
                except Exception as e:
                    self.log(f"[讀取失敗] {os.path.basename(txt)}：{e}")
                    continue

                basename = os.path.splitext(txt)[0]
                if fmt in ("csv", "both"):
                    csv_path = basename + ".csv"
                    try:
                        df.to_csv(csv_path, index=False, header=False, encoding='utf-8')
                        self.log(f"[已產生 CSV] {os.path.basename(csv_path)}")
                    except Exception as e:
                        self.log(f"[CSV 寫入失敗] {os.path.basename(csv_path)}：{e}")

                if fmt in ("xlsx", "both"):
                    xlsx_path = basename + ".xlsx"
                    try:
                        df.to_excel(xlsx_path, index=False, header=False, engine='openpyxl')
                        self.log(f"[已產生 XLSX] {os.path.basename(xlsx_path)}")
                    except Exception as e:
                        self.log(f"[XLSX 寫入失敗] {os.path.basename(xlsx_path)}：{e}")

        self.log("轉檔完成！")
        self.btn_start.config(state='normal')


if __name__ == "__main__":
    root = tk.Tk()
    app = ConverterApp(root)
    root.mainloop()
