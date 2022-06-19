class Params:
    def __init__(self):
        self.data_count = 0
        self.offline_mode = 0 # 0 if offline mode is generic; 1 for adverserial mode
        self.max_f1 = 0
        self.yesterday_f1 = 0
        self.record_id = 0
