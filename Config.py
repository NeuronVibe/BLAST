import os

class Config:
    def __init__(self):
        # self.file_directory = "F:/Spindle/my_code/dataset/MASS/input_data"
        self.file_directory = "F:/Spindle/my_code/dataset/DREAMS/input_data"
        self.signal_fs = 100  # 信号采样频率
        self.window = 20
        self.slide_window = 5
        self.signal_len = self.signal_fs * self.window
        self.min_iou = 0.2

        self.batch_size = 32
        self.epochs = 100
        self.chan_list = ['data']
        self.event_list = ["spindle"]

        self.save_event = False