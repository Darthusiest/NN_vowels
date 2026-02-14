import numpy as np
import matplotlib.pyplot as plt


class VowelNN:
    def __init__(self, batch_size=64, momentum=0.9, weight_decay=1e-4, early_stopping_patience=80):
        self.x_train = None
        self.y_train = None

        self.LR = 0.15
        self.batch_size = batch_size
        self.momentum = momentum #momentum for the update of the weights
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience

        self.input_size = 3
        self.hidden_size = 128
        self.hidden_size_2 = 64
        self.output_size = 12

        # He initialization for ReLU
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2 / self.input_size)
        self.B1 = np.zeros(self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.hidden_size_2) * np.sqrt(2 / self.hidden_size)
        self.B2 = np.zeros(self.hidden_size_2)
        self.W3 = np.random.randn(self.hidden_size_2, self.output_size) * np.sqrt(2 / self.hidden_size_2)
        self.B3 = np.zeros(self.output_size)

        # Momentum velocities, created to store the previous updates of the weights in arrays filled with 0's
        self.vW1 = np.zeros_like(self.W1)
        self.vB1 = np.zeros_like(self.B1)
        self.vW2 = np.zeros_like(self.W2)
        self.vB2 = np.zeros_like(self.B2)
        self.vW3 = np.zeros_like(self.W3)
        self.vB3 = np.zeros_like(self.B3)


        path = "bigdata.dat.txt"
        x_all, y_all = self.load_data(path)
        self.x_train, self.y_train, self.x_test, self.y_test = self.split_data(x_all, y_all, test_frac=0.2, seed=42)





    def load_data(self, path: str):
        x = []
        y = []

        vowels = {"ae": 0, "ah": 1, "aw": 2, "eh": 3, 
                  "er": 4, "ey": 5, "ei": 5, "ih": 6, "iy": 7, 
                  "oa": 8, "oo": 9, "uh": 10, "uw": 11}

        with open(path, 'r') as file:
            for line in file:
                data = line.strip().split() #remove whitespace and split into list
                

                if len(data) < 6:
                    continue #skip lines until we are getting the formants
                
                filename = data[0]
                if filename[0] not in ["m", "w", "b", "g"]:
                    continue #skip lines until we are getting the correct gender


                #get formants from data
                f1 = float(data[3])
                f2 = float(data[4])
                f3 = float(data[5])

                vowel = filename[3:5]

                x.append([f1, f2, f3])
                y.append(vowels[vowel])

        return np.array(x), np.array(y)




    #Stratified 80/20 train/test split so each vowel is represented in both sets.
    def split_data(self, x, y, test_frac=0.2, seed=42):
        np.random.seed(seed)
        train_idx, test_idx = [], []

        for class_id in range(self.output_size):
            mask = y == class_id
            indices = np.where(mask)[0]
            np.random.shuffle(indices)
            n_test = max(1, int(len(indices) * test_frac))
            n_train = len(indices) - n_test
            train_idx.extend(indices[:n_train])
            test_idx.extend(indices[n_train:])
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)
        return x[train_idx], y[train_idx], x[test_idx], y[test_idx]




    def process_data(self):
        """Standardize train and test data using train statistics only (no data leakage)."""
        x_train = self.x_train.astype(float)
        x_test = self.x_test.astype(float)

        # Compute mean and std from TRAIN only
        self.feature_mean = x_train.mean(axis=0) #mean of each formant set
        self.feature_std = x_train.std(axis=0) #standard deviation of each formant set (how much each set varies)
        self.feature_std[self.feature_std == 0] = 1.0 #replace 0s with 1s to avoid division by zero

        # Standardize both sets
        self.x_train = (x_train - self.feature_mean) / self.feature_std
        self.x_test = (x_test - self.feature_mean) / self.feature_std

        # Shuffle data so no bias in learning
        index = np.random.permutation(self.x_train.shape[0])
        self.x_train = self.x_train[index]
        self.y_train = self.y_train[index]





    def sigmoid(self, epoch, max_epochs, max_lr = 0.15, min_lr = 0.01, k = 5):
        middle = max_epochs / 2
        return min_lr + (max_lr - min_lr) / (1 + np.exp(k * (epoch - middle) / middle))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(np.float64)

    def sigmoid_activation(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        s = 1 / (1 + np.exp(-x))
        return s * (1 - s)



    def train_model(self, epochs=1000):
        self.process_data()
        self.loss_history = []
        self.accuracy_history = []
        n = self.x_train.shape[0]
        best_test_loss = np.inf
        epochs_without_improvement = 0

        for epoch in range(epochs):
            self.LR = self.sigmoid(epoch, epochs)
            # Shuffle training data each epoch
            perm = np.random.permutation(n)
            x_shuf = self.x_train[perm]
            y_shuf = self.y_train[perm]

            epoch_losses = []
            epoch_correct = 0
            epoch_total = 0

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                x = x_shuf[start:end]
                y = y_shuf[start:end]

                # Forward pass (ReLU hidden layers)
                weight_sum_1 = np.dot(x, self.W1) + self.B1
                output_H1_layer = self.relu(weight_sum_1)
                weight_sum_2 = np.dot(output_H1_layer, self.W2) + self.B2
                output_H2_layer = self.relu(weight_sum_2)
                scores = np.dot(output_H2_layer, self.W3) + self.B3
                probabilities = self.softmax(scores)

                N = y.shape[0]
                prevent_crash = 1e-12
                correct_probs = probabilities[np.arange(N), y]
                loss = -np.mean(np.log(correct_probs + prevent_crash))
                epoch_losses.append(loss)
                predictions = np.argmax(probabilities, axis=1)
                epoch_correct += np.sum(predictions == y)
                epoch_total += N

                self.back_propagation(x, y, probabilities, weight_sum_1, output_H1_layer, weight_sum_2, output_H2_layer)

            train_loss = np.mean(epoch_losses)
            train_acc = epoch_correct / epoch_total
            self.loss_history.append(train_loss)
            self.accuracy_history.append(train_acc)

            # Test loss for early stopping
            test_loss, _ = self._loss_and_accuracy(self.x_test, self.y_test)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epoch % 100 == 0:
                print(f"Epoch {epoch} - LR: {self.LR:.4f} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.2%} | Test loss: {test_loss:.4f}")

            if epochs_without_improvement >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (no test loss improvement for {self.early_stopping_patience} epochs)")
                break

        plt.plot(self.loss_history, label="Loss")
        plt.plot(self.accuracy_history, label="Accuracy")
        plt.legend()
        plt.show()
        self.print_vowel_accuracy(self.x_test, self.y_test, title_suffix="(Test set — unseen data)")





    def back_propagation(self, x, y, probabilities, weight_sum_1, output_H1_layer, weight_sum_2, output_H2_layer):
        N = x.shape[0]
        dScores = probabilities.copy()
        dScores[np.arange(N), y] -= 1
        dScores /= N

        dW3 = np.dot(output_H2_layer.T, dScores)
        dB3 = np.sum(dScores, axis=0)
        dOutput_H2_layer = np.dot(dScores, self.W3.T)
        dWeight_sum_2 = dOutput_H2_layer * self.relu_derivative(weight_sum_2)

        dW2 = np.dot(output_H1_layer.T, dWeight_sum_2)
        dB2 = np.sum(dWeight_sum_2, axis=0)
        dOutput_H1_layer = np.dot(dWeight_sum_2, self.W2.T)
        dWeight_sum_1 = dOutput_H1_layer * self.relu_derivative(weight_sum_1)

        dW1 = np.dot(x.T, dWeight_sum_1)
        dB1 = np.sum(dWeight_sum_1, axis=0)

        # L2 regularization gradient
        dW1 += self.weight_decay * self.W1
        dW2 += self.weight_decay * self.W2
        dW3 += self.weight_decay * self.W3

        # Momentum updates
        self.vW1 = self.momentum * self.vW1 + dW1
        self.vB1 = self.momentum * self.vB1 + dB1
        self.vW2 = self.momentum * self.vW2 + dW2
        self.vB2 = self.momentum * self.vB2 + dB2
        self.vW3 = self.momentum * self.vW3 + dW3
        self.vB3 = self.momentum * self.vB3 + dB3

        self.W1 -= self.LR * self.vW1
        self.B1 -= self.LR * self.vB1
        self.W2 -= self.LR * self.vW2
        self.B2 -= self.LR * self.vB2
        self.W3 -= self.LR * self.vW3
        self.B3 -= self.LR * self.vB3





    def softmax(self, scores):
        scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def _loss_and_accuracy(self, x, y):
        """Forward pass with ReLU; returns cross-entropy loss and accuracy."""
        weight_sum_1 = np.dot(x, self.W1) + self.B1
        output_H1_layer = self.relu(weight_sum_1)
        weight_sum_2 = np.dot(output_H1_layer, self.W2) + self.B2
        output_H2_layer = self.relu(weight_sum_2)
        scores = np.dot(output_H2_layer, self.W3) + self.B3
        probabilities = self.softmax(scores)
        N = y.shape[0]
        prevent_crash = 1e-12
        correct_probs = probabilities[np.arange(N), y]
        loss = -np.mean(np.log(correct_probs + prevent_crash))
        predictions = np.argmax(probabilities, axis=1)
        accuracy = np.mean(predictions == y)
        return loss, accuracy





    def print_vowel_accuracy(self, x, y, title_suffix=""):
        """Display per-vowel correct/total accuracy in a figure."""
        output_H1_layer = self.relu(np.dot(x, self.W1) + self.B1)
        output_H2_layer = self.relu(np.dot(output_H1_layer, self.W2) + self.B2)
        scores = np.dot(output_H2_layer, self.W3) + self.B3
        probabilities = self.softmax(scores)
        predictions = np.argmax(probabilities, axis=1)

        vowel_names = ["ae", "ah", "aw", "eh", "er", "ey", "ih", "iy", "oa", "oo", "uh", "uw"]

        correct_counts = []
        total_counts = []
        labels = []
        for i, name in enumerate(vowel_names):
            mask = y == i
            count = int(np.sum(mask))
            correct = int(np.sum((predictions == y) & mask))
            if count > 0:
                correct_counts.append(correct)
                total_counts.append(count)
                labels.append(f"{name}\n({correct}/{count})")

        total_correct = sum(correct_counts)
        total_count = sum(total_counts)

        # Display as a bar chart figure
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(labels))
        bars = ax.bar(x_pos, [c/t for c, t in zip(correct_counts, total_counts)], color='steelblue', edgecolor='navy')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Accuracy")
        title = f"Per-vowel accuracy — Total: {total_correct}/{total_count} correct"
        if title_suffix:
            title += f" {title_suffix}"
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()





def main():
    model = VowelNN()
    model.train_model()

    print("Training complete")
    print("Train — Loss:", model.loss_history[-1], "| Accuracy:", model.accuracy_history[-1])
    
    
    # Test accuracy (generalization)
    output_H1_layer = model.relu(np.dot(model.x_test, model.W1) + model.B1)
    output_H2_layer = model.relu(np.dot(output_H1_layer, model.W2) + model.B2)
    preds = np.argmax(model.softmax(np.dot(output_H2_layer, model.W3) + model.B3), axis=1)
    test_acc = np.mean(preds == model.y_test)
    print("Test  — Accuracy:", f"{test_acc:.2%}", "(unseen data)")

if __name__ == "__main__":
    main()