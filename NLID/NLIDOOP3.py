import numpy as np
import matplotlib.pyplot as plt

class RecurrenceAnalysis:
    def __init__(self, data, m, tau):
        """
        初始化 RecurrenceAnalysis 对象。
        :param data: 输入的一维时间序列
        :param m: 嵌入维度
        :param tau: 时间延迟
        """
        self.data = np.asarray(data, dtype=np.float64)
        self.m = m
        self.tau = tau
        self.phase_space = None

    def reconstruct_phase_space(self):
        """
        重构相空间。
        """
        L = len(self.data)
        M = L - (self.m - 1) * self.tau
        self.phase_space = np.zeros((M, self.m), dtype=np.float64)
        for p in range(self.m):
            self.phase_space[:, p] = self.data[np.arange(0, M) + p * self.tau]
        return self.phase_space

    @staticmethod
    def compute_reconstruction_matrix(phase_space, threshold=None, threshold_type="dynamic"):
        """
        计算重建矩阵 R(i, j)。
        :param phase_space: 相空间矩阵
        :param threshold: 静态或动态的阈值
        :param threshold_type: "static" 或 "dynamic"
        :return: 重建矩阵或二值化矩阵
        """
        squared_norms = np.sum(phase_space**2, axis=1, keepdims=True)
        distance_matrix = np.sqrt(
            np.maximum(0, squared_norms + squared_norms.T - 2 * np.dot(phase_space, phase_space.T))
        )

        if threshold_type == "static" and threshold is not None:
            return (distance_matrix <= threshold).astype(int)
        elif threshold_type == "dynamic" and threshold is not None:
            dTH = (np.max(distance_matrix) - np.min(distance_matrix)) * threshold
            return (distance_matrix <= dTH).astype(int)

        return distance_matrix


    @staticmethod
    def visualize_recurrence_plot(matrix, title, xlabel, ylabel):
        """
        可视化重现图。
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(matrix, cmap='gray_r', origin='lower')
        plt.title(title, fontsize=14)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.colorbar()
        plt.show()

    @staticmethod
    def calculate_nlid(AR_EEG1_BW, AR_EEG2_BW):
        """
        计算 NLID 指标。
        """
        N = AR_EEG1_BW.shape[1]

        # 初始化为浮点数组
        NLID_YX = np.zeros(N, dtype=np.float32)
        NLID_XY = np.zeros(N, dtype=np.float32)

        # 批量矩阵操作
        IP = AR_EEG1_BW * AR_EEG2_BW
        number_of_1 = np.sum(IP, axis=0, dtype=np.float32)
        number_of_EEG1 = np.sum(AR_EEG1_BW, axis=0, dtype=np.float32)
        number_of_EEG2 = np.sum(AR_EEG2_BW, axis=0, dtype=np.float32)

        # 避免类型错误，确保输出类型为浮点数
        NLID_YX = np.divide(number_of_1, number_of_EEG1, where=number_of_EEG1 > 0, out=NLID_YX)
        NLID_XY = np.divide(number_of_1, number_of_EEG2, where=number_of_EEG2 > 0, out=NLID_XY)

        NLID_YX_avg = np.mean(NLID_YX)
        NLID_XY_avg = np.mean(NLID_XY)

        return NLID_XY_avg, NLID_YX_avg

