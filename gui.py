import tkinter as tk
import random
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from network import Layer , NeuralNetwork , sigmoid 

class NeuralNetworkGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("BARU Weka")

        self.create_widgets()

        self.figure, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        self.loss_line, = self.ax1.plot([], [], label='Loss', color='blue')
        self.acc_line, = self.ax2.plot([], [], label='Accuracy', color='green')

        self.ax1.set_ylabel('Train Loss', color='blue')
        self.ax1.set_xlabel('Epochs')

        self.ax2.set_xlabel('Epochs')
        self.ax2.set_ylabel('Test Accuracy', color='green')

        self.ax1.legend(loc='upper left')
        self.ax2.legend(loc='upper left')

        self.canvas = FigureCanvasTkAgg(self.figure, master=master)
        self.canvas.get_tk_widget().grid(row=6, column=0, columnspan=2, sticky="nsew")
        self.animation_interval = 5

    def create_widgets(self):
        ttk.Label(self.master, text="Learning Rate:").grid(row=0, column=0, pady=5, padx=10, sticky="w")
        self.learning_rate_entry = ttk.Entry(self.master)
        self.learning_rate_entry.insert(0, "0.1")
        self.learning_rate_entry.grid(row=0, column=1, pady=5, padx=10, sticky="w")

        ttk.Label(self.master, text="Batch Size:").grid(row=1, column=0, pady=5, padx=10, sticky="w")
        self.batch_size_entry = ttk.Entry(self.master)
        self.batch_size_entry.insert(0, "32")
        self.batch_size_entry.grid(row=1, column=1, pady=5, padx=10, sticky="w")

        ttk.Label(self.master, text="Epochs:").grid(row=2, column=0, pady=5, padx=10, sticky="w")
        self.epochs_entry = ttk.Entry(self.master)
        self.epochs_entry.insert(0, "100")
        self.epochs_entry.grid(row=2, column=1, pady=5, padx=10, sticky="w")

        ttk.Label(self.master, text="Momentum:").grid(row=3, column=0, pady=5, padx=10, sticky="w")
        self.momentum_entry = ttk.Entry(self.master)
        self.momentum_entry.insert(0, "0.9")
        self.momentum_entry.grid(row=3, column=1, pady=5, padx=10, sticky="w")

        ttk.Label(self.master, text="Select Optimizer:").grid(row=4, column=0, pady=5, padx=10, sticky="w")
        self.optimizer_var = tk.StringVar()
        self.optimizer_combobox = ttk.Combobox(self.master, textvariable=self.optimizer_var, values=["SDG" ,"Momentum"])
        self.optimizer_combobox.grid(row=4, column=1, pady=5, padx=10, sticky="w")
        self.optimizer_combobox.set("SDG")

        ttk.Label(self.master, text="Select Dataset:").grid(row=5, column=0, pady=5, padx=10, sticky="w")
        self.dataset_var = tk.StringVar()
        self.dataset_combobox = ttk.Combobox(self.master, textvariable=self.dataset_var, values=["MNIST"])
        self.dataset_combobox.grid(row=5, column=1, pady=5, padx=10, sticky="w")
        self.dataset_combobox.set("MNIST")


        ttk.Button(self.master, text="Train Model", command=self.train_model).grid(row=5, column=1, columnspan=2, pady=20)

    def one_hot_encode(self, labels, num_classes):
        labels = np.array(labels, dtype=np.int32)
        encoded_labels = np.zeros((len(labels), num_classes))

        for i in range(len(labels)):
            encoded_labels[i, labels[i]] = 1
        return encoded_labels

    def load_dataset(self, dataset_name, test_size):
        if dataset_name == "MNIST":
            dataset = load_digits()
        else:
            raise ValueError("Invalid dataset name")

        X = dataset.data
        y = dataset.target.reshape(-1, 1)
        X = X / np.max(X)
        y_onehot = self.one_hot_encode(y, len(np.unique(dataset.target)))
        X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=test_size)
        return X_train, X_test, y_train, y_test

    
    def update_metrics(self, epoch, train_losses, test_losses, train_accuracies, test_accuracies):
        epochs = np.arange(epoch + 1)

        self.loss_line.set_xdata(epochs)
        self.loss_line.set_ydata(train_losses)

        self.acc_line.set_xdata(epochs)
        self.acc_line.set_ydata(test_accuracies)

        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.canvas.draw()

    def train_network_epoch(self, nn, X_train, y_train, X_test, y_test, epoch):
        for i in range(0, len(X_train), int(self.batch_size_entry.get())):
            batch_inputs = X_train[i:i + int(self.batch_size_entry.get())]
            batch_labels = y_train[i:i + int(self.batch_size_entry.get())]
            prediction = nn.forward(batch_inputs)
            nn.backward(batch_labels)

        train_loss =  np.sum((prediction - batch_labels) ** 2)
        test_prediction = nn.predict(X_test)
        test_loss = np.sum((test_prediction - y_test) ** 2)
        train_accuracy = np.mean(np.argmax(prediction, axis=1) == np.argmax(batch_labels, axis=1))
        test_accuracy = np.mean(np.argmax(test_prediction, axis=1) == np.argmax(y_test, axis=1))

        return train_loss, test_loss, train_accuracy, test_accuracy
    def test_on_samples(self,X_test  , y_test , nn,test_count):

        predictions = []
        for _ in range(test_count):  
            sample_index = random.randint(0, len(X_test) - 1)
            sample_data = X_test[sample_index, :].reshape(1, -1)
            sample_label = y_test[sample_index, :].reshape(1, -1)
            prediction = nn.predict(sample_data)
            predicted_class = np.argmax(prediction)
            true_class = np.argmax(sample_label)
            predictions.append(f"True Class: {true_class}\t\tPredicted Class: {predicted_class}\n")
        return predictions

    def train_model(self):
        learning_rate = float(self.learning_rate_entry.get())
        batch_size = int(self.batch_size_entry.get())
        epochs = int(self.epochs_entry.get())
        dataset_name = self.dataset_var.get()

        X_train, X_test, y_train, y_test = self.load_dataset(dataset_name, test_size=0.2)

        nn = NeuralNetwork(learning_rate=learning_rate, batch_size=batch_size , optimizer=self.optimizer_var.get() , momentum=float(self.momentum_entry.get()))
        nn.add_layer(Layer(input_size=X_train.shape[1], output_size=64, activation_function=sigmoid))
        nn.add_layer(Layer(input_size=64, output_size=y_train.shape[1], activation_function=sigmoid))

        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []
        
        def train_one_epoch_and_update_metrics(epoch):
            nonlocal train_losses, test_losses, train_accuracies, test_accuracies
            train_loss, test_loss, train_accuracy, test_accuracy = self.train_network_epoch(nn, X_train, y_train, X_test, y_test, epoch)

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)

            self.update_metrics(epoch, train_losses,test_losses, train_accuracies, test_accuracies)

            if epoch < epochs - 1:
                self.master.after(self.animation_interval, train_one_epoch_and_update_metrics, epoch + 1)
            else:
                self.show_statistics(train_losses,  test_accuracies)
                test_results = self.test_on_samples(X_test, y_test,nn, test_count=20)
                self.show_test_results(test_results)

        train_one_epoch_and_update_metrics(0)

    def show_test_results(self, test_results):
        test_results_window = tk.Toplevel(self.master)
        test_results_window.title("Test Results")

        ttk.Label(test_results_window, text="Test Results:").pack(pady=5)
        test_results_text = tk.Text(test_results_window, height=50, width=50)
        test_results_text.pack()

        test_results_text.insert(tk.END, "True Class\t\tPredicted Class\n")
        test_results_text.insert(tk.END, "-" * 34 + "\n")

        for prediction  in test_results:
            test_results_text.insert(tk.END, prediction) 


    def show_statistics(self, train_losses, test_accuracies):
        statistics_window = tk.Toplevel(self.master)
        statistics_window.title("Training Statistics")

        ttk.Label(statistics_window, text="Train Loss:").pack(pady=5)
        train_losses_text = tk.Text(statistics_window, height=2, width=30)
        train_losses_text.pack()
        train_losses_text.insert(tk.END, str(train_losses[len(train_losses) -1]))

        ttk.Label(statistics_window, text="Test Accuracy:").pack(pady=5)
        test_accuracies_text = tk.Text(statistics_window, height=2, width=30)
        test_accuracies_text.pack()
        test_accuracies_text.insert(tk.END, str(test_accuracies[len(test_accuracies) -1]))

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetworkGUI(root)
    root.mainloop()

