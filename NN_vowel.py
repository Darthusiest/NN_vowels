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

    def train_model(self, epochs: int):
        self.loss_history = []
        self.accuracy_history = []
        
        for epoch in range(epochs):

            #forward pass

            #compute loss

            #compute accuracy

            #backpropagation

            #update weights and biases



        #after training, plot the loss and accuracy history
        plt.plot(self.loss_history)
        plt.plot(self.accuracy_history)
        plt.show()


    def softmax(self, z):
        pass




def main():
    pass

if __name__ == "__main__":
    main()