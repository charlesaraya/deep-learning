class EarlyStopping:
    def __init__(self, patience = 5, threshold = 0.01, verbose = False):
        self.patience = patience
        self.threshold = threshold
        self.verbose = verbose

        self.patience_left = self.patience
        self.best_value = None

    def __call__(self, current_metric) -> bool:

        if self.best_value is None or self.best_value - current_metric >= self.threshold:
            self.best_value = current_metric
            self.patience_left = self.patience
            return False

        if current_metric > self.best_value or self.best_value - current_metric < self.threshold:
            if self.patience_left == 0:
                return True
            self.patience_left -= 1
            if self.verbose:
                if current_metric > self.best_value:
                    reason = "↓"
                if self.best_value - current_metric > 0 and self.best_value - current_metric < self.threshold:
                    reason = "θ"
                print(f"Pat[{reason}]({self.patience_left}/{self.patience}) ", end="")
        return False

if __name__ == '__main__':
    import numpy as np
    np.random.seed(42)

    class Perceptron:
        def __init__(self, indim, lr = 0.01):
            self.w = np.random.uniform(-1, 1, (indim,)) #* 0.1
            self.b = 0.0
            self.lr = lr

        def __call__(self, x):
            return self.w @ x + self.b

        def backward(self, grad):
            self.dw = grad * self.w
            self.db = grad
            self.w -= self.lr * self.dw
            self.b -= self.lr * self.db

    # Model
    epochs = 100
    indim = 10
    p = Perceptron(indim, 0.1)
    # Training config
    patience = 5
    early_stopping = EarlyStopping(patience, 0.001, verbose=True)

    # train data
    input_data = np.random.randint(-10, 10, size=(indim,))
    # Norm
    input_data = (input_data - np.mean(input_data)) // np.std(input_data)
    y = 1 # np.random.randint(2)

    # train
    for i, epoch in enumerate(range(epochs)):
        logits = p(input_data)
        loss = np.sum((y - logits)**2)
        if early_stopping(loss):
            print(f"Early Stopping triggered at epoch {epoch}")
            break
        p.backward(loss)
        print(f"{loss:.4f}", end=" ")
