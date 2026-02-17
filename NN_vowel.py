import numpy as np
import matplotlib.pyplot as plt


class VowelNN:
    def __init__(self, batch_size=128, momentum=0.85, weight_decay=1e-5, early_stopping_patience=120, max_grad_norm=1.0):
        self.x_train = None
        self.y_train = None

        self.LR = 0.03
        self.batch_size = batch_size
        self.momentum = momentum #momentum for the update of the weights
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.max_grad_norm = max_grad_norm

        self.input_size = 31  # 7 base (F1,F2,F3,ratios,diffs) + 24 trajectory (F1,F2,F3 at 10%..80%)
        self.hidden_size = 256   # slightly larger for 12-class; helps reach ~90%
        self.hidden_size_2 = 128
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
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = self.split_data(
            x_all, y_all, test_frac=0.2, val_frac=0.15, seed=42
        )





    def load_data(self, path: str):
        x = []
        y = []

        vowels = {"ae": 0, "ah": 1, "aw": 2, "eh": 3, 
                  "er": 4, "ey": 5, "ei": 5, "ih": 6, "iy": 7, 
                  "oa": 8, "oo": 9, "uh": 10, "uw": 11}

        with open(path, 'r') as file:
            for line in file:
                data = line.strip().split() #remove whitespace and split into list
                

                if len(data) < 30:
                    continue  # need cols 1–30 for trajectory (F1,F2,F3 at 10%..80%)
                
                filename = data[0]
                if filename[0] not in ["m", "w", "b", "g"]:
                    continue #skip lines until we are getting the correct gender

                vowel = filename[3:5]
                # get formants from data
                f1 = float(data[3])
                f2 = float(data[4])
                f3 = float(data[5])

                # avoid division by zero: treat 0 as 1 for ratio computation
                f1_safe = f1 if f1 != 0 else 1.0
                f2_safe = f2 if f2 != 0 else 1.0
                f3_safe = f3 if f3 != 0 else 1.0

                ratio_f1_f2 = f1_safe / f2_safe
                ratio_f3_f2 = f3_safe / f2_safe
                diff_f2_f1 = f2 - f1
                diff_f3_f2 = f3 - f2

                # trajectory: F1,F2,F3 at 10%,20%,...,80% (cols 7–30 → indices 6–29)
                trajectory = [float(data[i]) for i in range(6, 30)]
                # 31 features: 7 base + 24 trajectory
                x.append([f1, f2, f3, ratio_f1_f2, ratio_f3_f2, diff_f2_f1, diff_f3_f2] + trajectory)
                y.append(vowels[vowel])

        return np.array(x), np.array(y)




    def split_data(self, x, y, test_frac=0.2, val_frac=0.15, seed=42):
        """Stratified train/val/test split so each vowel is represented in all sets."""
        np.random.seed(seed)
        train_idx, val_idx, test_idx = [], [], []

        for class_id in range(self.output_size):
            mask = y == class_id
            indices = np.where(mask)[0]
            np.random.shuffle(indices)
            n = len(indices)
            n_test = max(1, int(n * test_frac))
            n_val = max(1, int(n * val_frac))
            n_train = n - n_test - n_val
            if n_train < 1:
                n_train = 1
                n_val = min(n_val, n - 2)
                n_test = n - n_train - n_val
            train_idx.extend(indices[:n_train])
            val_idx.extend(indices[n_train : n_train + n_val])
            test_idx.extend(indices[n_train + n_val :])
        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)
        test_idx = np.array(test_idx)
        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)
        np.random.shuffle(test_idx)
        return (
            x[train_idx], y[train_idx],
            x[val_idx], y[val_idx],
            x[test_idx], y[test_idx],
        )




    def process_data(self):
        """Standardize train, val, and test using train statistics only (no data leakage)."""
        #convert the data to floats
        x_train = self.x_train.astype(float)
        x_val = self.x_val.astype(float)
        x_test = self.x_test.astype(float)

        #calculate the mean and standard deviation of the data
        self.feature_mean = x_train.mean(axis=0)
        self.feature_std = x_train.std(axis=0)
        self.feature_std[self.feature_std == 0] = 1.0

        #standardize the data
        self.x_train = (x_train - self.feature_mean) / self.feature_std
        self.x_val = (x_val - self.feature_mean) / self.feature_std
        self.x_test = (x_test - self.feature_mean) / self.feature_std

        #shuffle the data
        index = np.random.permutation(self.x_train.shape[0])
        self.x_train = self.x_train[index]
        self.y_train = self.y_train[index]





    def sigmoid(self, epoch, max_epochs, max_lr=0.04, min_lr=0.0005, k=4):
        """LR schedule: start higher, decay more gently (k=4) so model can refine longer."""
        middle = max_epochs / 2
        return min_lr + (max_lr - min_lr) / (1 + np.exp(k * (epoch - middle) / middle))

    def relu(self, x):
        #apply the relu function to the input
        return np.maximum(0, x)

    def relu_derivative(self, x):
        #calculate the derivative of the relu function
        return (x > 0).astype(np.float64)

    def sigmoid_activation(self, x):
        #apply the sigmoid function to the input
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        #calculate the derivative of the sigmoid function
        s = 1 / (1 + np.exp(-x))
        return s * (1 - s)



    def train_model(self, epochs=2000):
        self.process_data()
        self.loss_history = []
        self.accuracy_history = []
        self.val_loss_history = []
        self.val_accuracy_history = []
        n = self.x_train.shape[0]
        best_val_loss = np.inf
        epochs_without_improvement = 0

        for epoch in range(epochs):
            self.LR = self.sigmoid(epoch, epochs)

            #Shuffle the data
            perm = np.random.permutation(n)
            x_shuf = self.x_train[perm]
            y_shuf = self.y_train[perm]

            #Variables to store the loss and accuracy for the epoch
            epoch_losses = []
            epoch_correct = 0
            epoch_total = 0

            #Batch training
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                x = x_shuf[start:end]
                y = y_shuf[start:end]

                #Forward pass
                #Hidden layer 1
                weight_sum_1 = np.dot(x, self.W1) + self.B1
                output_H1_layer = self.relu(weight_sum_1)

                #Hidden layer 2
                weight_sum_2 = np.dot(output_H1_layer, self.W2) + self.B2
                output_H2_layer = self.relu(weight_sum_2)

                #Output layer
                scores = np.dot(output_H2_layer, self.W3) + self.B3
                probabilities = self.softmax(scores)

                #Calculate the loss and accuracy
                N = y.shape[0]
                prevent_crash = 1e-12
                correct_probs = probabilities[np.arange(N), y]

                #cross-entropy loss
                loss = -np.mean(np.log(correct_probs + prevent_crash))
                epoch_losses.append(loss)
                predictions = np.argmax(probabilities, axis=1)

                #predictions is an array of the predicted classes for each sample
                #calculate the accuracy
                epoch_correct += np.sum(predictions == y)
                epoch_total += N

                #Backpropagation
                self.back_propagation(x, y, probabilities, weight_sum_1, output_H1_layer, weight_sum_2, output_H2_layer)

            #Calculate the average loss and accuracy for the epoch
            train_loss = np.mean(epoch_losses)
            train_acc = epoch_correct / epoch_total
            self.loss_history.append(train_loss)
            self.accuracy_history.append(train_acc)

            #Calculate the loss and accuracy for the validation set
            val_loss, val_acc = self._loss_and_accuracy(self.x_val, self.y_val)
            self.val_loss_history.append(val_loss)
            self.val_accuracy_history.append(val_acc)

            #If the validation loss is lower than the best validation loss 
            #update the best validation loss and reset the epochs without improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                #If the validation loss is not lower than the best validation loss
                #Increment the number of epochs without improvement
                epochs_without_improvement += 1


            #Print the loss and accuracy for the epoch every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}")
                print(f"  Learning rate:  {self.LR:.4f}")
                print(f"  Train — loss: {train_loss:.4f}   acc: {train_acc:.2%}")
                print(f"  Val   — loss: {val_loss:.4f}   acc: {val_acc:.2%}")
                print()

            if epochs_without_improvement >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (no val loss improvement for {self.early_stopping_patience} epochs)")
                break

        # Popup 1: Train loss & accuracy
        plt.figure()
        plt.plot(self.loss_history, label="Loss")
        plt.plot(self.accuracy_history, label="Accuracy")
        plt.legend()
        plt.xlabel("Epoch")
        plt.show()

        # Popup 2: Per-vowel accuracy bar chart
        self.print_vowel_accuracy(self.x_test, self.y_test, title_suffix="(Test set — unseen data)")

        # Popup 3: Training vs validation loss & accuracy (separate figure)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        epochs_range = range(len(self.loss_history))
        ax1.plot(epochs_range, self.loss_history, label="Train loss", color="C0")
        ax1.plot(epochs_range, self.val_loss_history, label="Val loss", color="C1")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.plot(epochs_range, self.accuracy_history, label="Train acc", color="C0")
        ax2.plot(epochs_range, self.val_accuracy_history, label="Val acc", color="C1")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()





    def back_propagation(self, x, y, probabilities, weight_sum_1, output_H1_layer, weight_sum_2, output_H2_layer):
        N = x.shape[0]
        
        
        #dScores is an array of the derivatives of the scores for each sample
        dScores = probabilities.copy()
        
        #subtract 1 from the score for the correct class
        dScores[np.arange(N), y] -= 1
        dScores /= N


        #Gradient for the weights and biases of the output layer
        dW3 = np.dot(output_H2_layer.T, dScores)
        dB3 = np.sum(dScores, axis=0)
        dOutput_H2_layer = np.dot(dScores, self.W3.T)


        #Gradient for the weights and biases of the hidden layer 2
        dWeight_sum_2 = dOutput_H2_layer * self.relu_derivative(weight_sum_2)


        #Gradient for the weights and biases of the hidden layer 1
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

        # Gradient clipping (scale all gradients if total norm exceeds max_grad_norm)
        grads = [dW1, dB1, dW2, dB2, dW3, dB3]
        total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
        if total_norm > self.max_grad_norm:
            scale = self.max_grad_norm / total_norm
            dW1, dB1, dW2, dB2, dW3, dB3 = [g * scale for g in grads]

        # Momentum updates
        self.vW1 = self.momentum * self.vW1 + dW1
        self.vB1 = self.momentum * self.vB1 + dB1
        self.vW2 = self.momentum * self.vW2 + dW2
        self.vB2 = self.momentum * self.vB2 + dB2
        self.vW3 = self.momentum * self.vW3 + dW3
        self.vB3 = self.momentum * self.vB3 + dB3

        #Update the weights and biases with the momentum
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





    def forward_single(self, x):
        """Forward pass for a single sample. x: shape (31,) or (1, 31). Returns (x, h1, h2, probs)."""
        x = np.atleast_2d(x) #convert the input to a 2D array
        weight_sum_1 = np.dot(x, self.W1) + self.B1


        h1 = self.relu(weight_sum_1) #apply the relu function to the input
        weight_sum_2 = np.dot(h1, self.W2) + self.B2


        h2 = self.relu(weight_sum_2) #apply the relu function to the input
        scores = np.dot(h2, self.W3) + self.B3

        
        probs = self.softmax(scores) #apply the softmax function to the input
        return x.squeeze(), h1.squeeze(), h2.squeeze(), probs.squeeze()





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

    # Interactive activation visualization
    from nn_visualizer import launch_visualizer
    launch_visualizer(model, model.x_test, model.y_test)

if __name__ == "__main__":
    main()