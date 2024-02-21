import java.util.ArrayList;

public class NeuralNetwork {
    public static void main(String[] args) {
        // Input data
        double[][] inputData = {{1, 0}, {2, 1}, {0, 1}, {-2, 1}};
        double[] outputData = {1, 9, 1, 7};

        // Initial weights
        double w1 = 0.5;
        double w2 = 0.5;

        // Learning rate
        double learningRate = 0.01;

        // Training
        int epochs = 1000;
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;
            double dw1 = 0;
            double dw2 = 0;

            for (int i = 0; i < inputData.length; i++) {
                double x1 = inputData[i][0];
                double x2 = inputData[i][1];
                double y = outputData[i];

                // Forward pass
                double predictedOutput = w1 * x1 + w2 * x2;

                // Compute loss
                double loss = Math.pow((y - predictedOutput), 2);
                totalLoss += loss;

                // Backward pass
                double dLoss = 2 * (predictedOutput - y);
                dw1 += dLoss * (-x1);
                dw2 += dLoss * (-x2);
            }

            // Update weights
            w1 -= learningRate * dw1 / inputData.length;
            w2 -= learningRate * dw2 / inputData.length;

            // Print loss
            if (epoch % 100 == 0) {
                System.out.println("Epoch " + epoch + ", Loss: " + totalLoss / inputData.length);
            }
        }

        // Print final weights
        System.out.println("Final weights:");
        System.out.println("w1: " + w1);
        System.out.println("w2: " + w2);
    }
}
