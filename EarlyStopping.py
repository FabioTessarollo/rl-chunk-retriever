class EarlyStopping:
    def __init__(self, patience=5, delta_ratio=0.01):
        """
        Args:
            patience (int): how many epochs to wait after last improvement
            delta_ratio (float): minimum relative improvement required, 
                                 e.g. 0.01 = 1% improvement
        """
        self.patience = patience
        self.delta_ratio = delta_ratio
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        required_improvement = self.best_score * (1 + self.delta_ratio)

        if score < required_improvement:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop