
import numpy as np
from sklearn.datasets import load_iris , load_digits
from sklearn.model_selection import train_test_split

class CNN:
    def __init__(self) -> None:
        self.conv_kernel = None
    def relu(self,x , derivative = False):
        if derivative:
             return np.greater(x, 0).astype(int)
        return np.maximum(0 , x)

    def sigmoid(self,x , derivative = False):
        if derivative:
            return self.sigmoid(x)* (1-self.sigmoid(x))
        return 1 / (1 + np.exp(-x))


    def correlate2d(self,image , kernel, padding):

        image = np.pad(image, pad_width=padding, mode='constant', constant_values=0)
        image_height, image_width = image.shape
        kernel_height, kernel_width = kernel[0] , kernel[1]

        if self.conv_kernel == None:
            self.conv_kernel = np.random.randn(1, *kernel) * np.sqrt(2 / np.prod(kernel))

        output_height = image_height - kernel_height + 1
        output_width = image_width - kernel_width + 1
        result = np.zeros((output_height, output_width))
        for i in range(output_height):
            for j in range(output_width):
                result[i, j] = np.sum(image[i:i+kernel_height, j:j+kernel_width] * self.conv_kernel)
        return result


    def max_pooling(self,input , karnel_size):

        image_height, image_width = input.shape
        kernel_height , kernel_width =  karnel_size[0] , karnel_size[1]
        output_height = image_height - kernel_height + 1
        output_width = image_width - kernel_width + 1

        result = np.zeros((output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                result[i, j] = np.max(input[i:i+kernel_height, j:j+kernel_width])
        return result


    def flatten(self,input):
        return input.reshape(-1,1)

    def forward(self,inputs):
        corelate_result = self.correlate2d(inputs , (5,5) ,0)
        relu_result = self.relu(corelate_result)
        max_pooling_result = self.max_pooling(relu_result ,(3,3))
        flatten = self.flatten(max_pooling_result)
        node_result = np.sum(flatten * np.random.uniform(-1,1,size=(flatten.shape[1],1)) )
        node_sigmoid_result = self.sigmoid(node_result)
        return node_sigmoid_result
    def backward(self , error):
        pass




cnn = CNN()

inputs = np.random.uniform(0, 255, (28 ,28)).astype(np.uint8)/255

cnn_result = cnn.forward(inputs)
print(cnn_result)


    
