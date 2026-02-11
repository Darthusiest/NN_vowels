import numpy as np
import matplotlib.pyplot as plt
import torch


class VowelNN:
    def __init__(self):
        self.x_train = None
        self.y_train = None

        self.LR = 0.01 # 10% learning rate

        self.input_size = 3
        self.hidden_size = 32
        self.output_size = 12


        self.W1 = np.random.randn(3, 32) * 0.01
        self.B1 = np.zeros(32)

        self.W2 = np.random.randn(32, 12) * 0.01
        self.B2 = np.zeros(12)

        path = "bigdata.dat.txt"
        self.x_train, self.y_train = self.load_data(path)



    def load_data(self, path: str):
        x = []
        y = []

        vowels = {"ae": 0, "ah": 1, "aw": 2, "eh": 3, 
                  "er": 4, "ey": 5, "ih": 6, "iy": 7, 
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

    def process_data(self):
        pass


    def train_model(self, epochs = 100):
        self.loss_history = []
        self.accuracy_history = []
        
        x = self.x_train
        y = self.y_train
        for epoch in range(epochs): #run through the data_set 100 times 

            #Forward Pass
            #Input --> Hidden Layer 1
            weight_sum_1 = np.dot(x, self.W1) + self.B1
            output_H1_layer = np.maximum(0, weight_sum_1)

            #Hidden --> Output
            scores = np.dot(output_H1_layer, self.W2) + self.B2

            probabilities = self.softmax(scores)
            
            
            
            #Compute Loss
            N = y.shape[0]
            prevent_crash = 1e-12
            correct_probs = probabilities[np.arange(N), y]
            loss = -np.mean(np.log(correct_probs + prevent_crash))

            #compute accuracy
            predictions = np.argmax(probabilities, axis = 1) #pick highest prob vowel per row
            accuracy = np.mean(predictions == y) #T/F accuracy per row

            #backpropagation
            

            #update weights and biases

            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)


        #after training, plot the loss and accuracy history
        plt.plot(self.loss_history, label = "Loss")
        plt.plot(self.accuracy_history, label = "Accuracy")
        plt.legend()
        plt.show()

    def back_propagation(self, x, y, probabilities, weight_sum_1, output_H1_layer):
        N = x.shape[0]

        #dScores = dL/dScores 
        dScores = probabilities.copy()
        dScores[np.arange(N), y] -= 1
        dScores /= N

        #Gradient for W2 & B2
        dW2 = np.dot(output_H1_layer.T, dScores)
        dB2 = np.sum(dScores, axis = 0)

        #Backprop into hidden layer
        dHidden = np.dot(dScores, self.W2.T)

        #zero out where pre-activation was <= 0
        dHidden[weight_sum_1 <= 0] = 0

        #Gradient for W1 & B1
        dW1 = np.dot(x.T, dHidden)
        dB1 = np.sum(dHidden, axis = 0)

        #Update weights and biases
        self.W1 -= self.LR * dW1
        self.B1 -= self.LR * dB1
        self.W2 -= self.LR * dW2
        self.B2 -= self.LR * dB2

        return dW1, dB1, dW2, dB2



    def softmax(self, z):
        pass




def main():
    pass

if __name__ == "__main__":
    main()