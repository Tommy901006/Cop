import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
import ttkbootstrap as tb
from ttkbootstrap.constants import *
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from PyEMD import EEMD
from NLIDOOP3 import RecurrenceAnalysis
import os

# ===== Hilbert–Huang 分析函數 (EEMD) =====
def process_signal(sig, fs, t, label, fmax=50, n_freq_bins=200, logbox=None):
    eemd = EEMD()
    imfs = eemd.emd(sig, t)
    if logbox:
        logbox.insert(tk.END, f"{label}: 分解得到 {imfs.shape[0]} 個 IMF (忽略最後一筆)\n")

    imf_amps = []
    for idx, imf in enumerate(imfs[:-1], 1):
        analytic_signal = hilbert(imf)
        amplitude = np.abs(analytic_signal)
        inst_phase = np.unwrap(np.angle(analytic_signal))
        inst_freq = np.diff(inst_phase) / (2*np.pi) * fs
        power = amplitude[:-1]**2

        weighted_freq = np.sum(inst_freq * power) / np.sum(power)
        mean_amp = np.mean(amplitude)

        msg = f"{label}_IMF {idx}: 主頻率 ≈ {weighted_freq:.2f} Hz, 平均振幅 = {mean_amp:.3f}\n"
        if logbox: logbox.insert(tk.END, msg)

        imf_amps.append(amplitude)

    return imf_amps

# ===== 主程式：執行分析 =====
def run_analysis(x, y, fs, fmax, window_sec, m, tau, logbox, file_label=""):
    try:
        t = np.arange(len(x)) / fs

        # EEMD Hilbert–Huang
        X_imf_amps = process_signal(x, fs, t, "X", fmax=fmax, logbox=logbox)
        Y_imf_amps = process_signal(y, fs, t, "Y", fmax=fmax, logbox=logbox)

        # Recurrence Analysis
        window_size = window_sec * fs
        results = []
        for imf_idx, (amp_x, amp_y) in enumerate(zip(X_imf_amps, Y_imf_amps), 1):
            nlid_xy_list, nlid_yx_list = [], []
            for start in range(0, len(amp_x) - window_size, window_size):
                seg_x = amp_x[start:start+window_size]
                seg_y = amp_y[start:start+window_size]
                ra_x = RecurrenceAnalysis(seg_x, m=m, tau=tau)
                ra_y = RecurrenceAnalysis(seg_y, m=m, tau=tau)
                AR_HR_BW = RecurrenceAnalysis.compute_reconstruction_matrix(
                    ra_x.reconstruct_phase_space(), threshold=0.1, threshold_type="dynamic"
                )
                AR_RP_BW = RecurrenceAnalysis.compute_reconstruction_matrix(
                    ra_y.reconstruct_phase_space(), threshold=0.1, threshold_type="dynamic"
                )
                NLID_XY_avg, NLID_YX_avg = RecurrenceAnalysis.calculate_nlid(AR_HR_BW, AR_RP_BW)
                nlid_xy_list.append(NLID_XY_avg)
                nlid_yx_list.append(NLID_YX_avg)

            mean_xy = np.mean(nlid_xy_list) if nlid_xy_list else np.nan
            mean_yx = np.mean(nlid_yx_list) if nlid_yx_list else np.nan
            results.append([file_label, imf_idx, mean_xy, mean_yx])

            msg = f"{file_label} IMF {imf_idx}: 平均 NLID_XY = {mean_xy:.3f}, 平均 NLID_YX = {mean_yx:.3f}\n"
            logbox.insert(tk.END, msg)

        df_out = pd.DataFrame(results, columns=["File", "IMF", "Mean_NLID_XY", "Mean_NLID_YX"])
        return df_out

    except Exception as e:
        logbox.insert(tk.END, f"❌ 錯誤 ({file_label}): {str(e)}\n", "error")
        return pd.DataFrame()

# ===== GUI 按鈕操作 =====
def choose_folder():
    path = filedialog.askdirectory()
    file_var.set(path)
    if path:
        logbox.insert(tk.END, f"📂 已選擇資料夾: {path}\n")

def run_from_folder():
    try:
        folder_path = file_var.get()
        col_x = col_x_var.get()
        col_y = col_y_var.get()
        fmax = float(fmax_var.get())
        window_sec = int(window_var.get())
        fs = int(fs_var.get())
        m = int(m_var.get())
        tau = int(tau_var.get())

        if not folder_path:
            logbox.insert(tk.END, "❌ 錯誤: 請選擇資料夾\n", "error")
            return

        all_dfs = []
        for fname in os.listdir(folder_path):
            if fname.endswith(".xlsx"):
                fpath = os.path.join(folder_path, fname)
                try:
                    df = pd.read_excel(fpath)
                    if col_x not in df.columns or col_y not in df.columns:
                        logbox.insert(tk.END, f"⚠️ {fname} 缺少指定欄位，跳過\n", "error")
                        continue

                    x = df[col_x].values
                    y = df[col_y].values

                    logbox.insert(tk.END, f"▶ 正在分析 {fname}...\n")
                    df_result = run_analysis(x, y, fs, fmax, window_sec, m, tau, logbox, file_label=fname)
                    if not df_result.empty:
                        all_dfs.append(df_result)

                except Exception as e:
                    logbox.insert(tk.END, f"❌ 錯誤處理 {fname}: {str(e)}\n", "error")

        if all_dfs:
            df_all = pd.concat(all_dfs, ignore_index=True)

            # 橫向展開
            df_pivot = df_all.pivot_table(
                index="File",
                columns="IMF",
                values=["Mean_NLID_XY", "Mean_NLID_YX"]
            )

            # 改欄位名稱格式 → Mean_NLID_XY_IMF1 ...
            df_pivot.columns = [f"{metric}_IMF{imf}" for metric, imf in df_pivot.columns]

            df_pivot.reset_index(inplace=True)
            df_pivot.to_csv("All_Files_IMF_NLID_mean.csv", index=False, encoding="utf-8-sig")

            logbox.insert(tk.END, "✅ 全部檔案分析完成！已輸出 All_Files_IMF_NLID_mean.csv\n", "success")
        else:
            logbox.insert(tk.END, "⚠️ 沒有成功分析的檔案\n", "error")

    except Exception as e:
        logbox.insert(tk.END, f"❌ 錯誤: {str(e)}\n", "error")

# ===== GUI 介面 =====
root = tb.Window(themename="cosmo")
root.title("EEMD + Recurrence Analysis 工具 (資料夾版)")
root.geometry("750x550")

file_var = tk.StringVar()
col_x_var = tk.StringVar()
col_y_var = tk.StringVar()
fmax_var = tk.StringVar(value="50")
window_var = tk.StringVar(value="1")
fs_var = tk.StringVar(value="1000")
m_var = tk.StringVar(value="3")
tau_var = tk.StringVar(value="1")

frm = ttk.Frame(root, padding=10)
frm.pack(fill=X)

ttk.Label(frm, text="資料夾:").grid(row=0, column=0, sticky=W, padx=5, pady=5)
ttk.Entry(frm, textvariable=file_var, width=40).grid(row=0, column=1, padx=5)
ttk.Button(frm, text="選擇資料夾", command=choose_folder, bootstyle=PRIMARY).grid(row=0, column=2, padx=5)

ttk.Label(frm, text="X 欄位:").grid(row=1, column=0, sticky=W, padx=5, pady=5)
col_x_menu = ttk.Entry(frm, textvariable=col_x_var)
col_x_menu.grid(row=1, column=1, padx=5)

ttk.Label(frm, text="Y 欄位:").grid(row=2, column=0, sticky=W, padx=5, pady=5)
col_y_menu = ttk.Entry(frm, textvariable=col_y_var)
col_y_menu.grid(row=2, column=1, padx=5)

ttk.Label(frm, text="頻率上限 fmax (Hz):").grid(row=3, column=0, sticky=W, padx=5, pady=5)
ttk.Entry(frm, textvariable=fmax_var).grid(row=3, column=1, padx=5)

ttk.Label(frm, text="窗口大小 (秒):").grid(row=4, column=0, sticky=W, padx=5, pady=5)
ttk.Entry(frm, textvariable=window_var).grid(row=4, column=1, padx=5)

ttk.Label(frm, text="取樣率 fs (Hz):").grid(row=5, column=0, sticky=W, padx=5, pady=5)
ttk.Entry(frm, textvariable=fs_var).grid(row=5, column=1, padx=5)

ttk.Label(frm, text="嵌入維度 m:").grid(row=6, column=0, sticky=W, padx=5, pady=5)
ttk.Entry(frm, textvariable=m_var).grid(row=6, column=1, padx=5)

ttk.Label(frm, text="延遲 tau:").grid(row=7, column=0, sticky=W, padx=5, pady=5)
ttk.Entry(frm, textvariable=tau_var).grid(row=7, column=1, padx=5)

ttk.Button(frm, text="開始分析 (資料夾)", command=run_from_folder, bootstyle=SUCCESS).grid(row=8, column=1, pady=10)

logbox = scrolledtext.ScrolledText(root, height=15, wrap=tk.WORD)
logbox.pack(fill=BOTH, expand=True, padx=10, pady=10)

logbox.tag_config("error", foreground="red")
logbox.tag_config("success", foreground="green")

root.mainloop()
