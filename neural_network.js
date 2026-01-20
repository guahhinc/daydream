class ActivationFunction {
    constructor(func, dfunc) {
        this.func = func;
        this.dfunc = dfunc;
    }
}

let sigmoid = new ActivationFunction(
    x => 1 / (1 + Math.exp(-x)),
    y => y * (1 - y)
);

let relu = new ActivationFunction(
    x => Math.max(0, x),
    y => y > 0 ? 1 : 0
);

class NeuralNetwork {
    // layer_sizes: array, e.g. [2, 4, 1] means 2 inputs, 4 hidden, 1 output
    constructor(layer_sizes) {
        this.layer_sizes = layer_sizes;
        this.weights = [];
        this.biases = [];
        this.learning_rate = 0.1;

        // Using ReLU for hidden layers, Sigmoid for output
        this.activation_hidden = relu;
        this.activation_output = sigmoid;

        // Initialize weights and biases
        for (let i = 0; i < layer_sizes.length - 1; i++) {
            let n_in = layer_sizes[i];
            let n_out = layer_sizes[i + 1];

            let weight = new Matrix(n_out, n_in);

            // He Initialization: Gaussian random * sqrt(2 / input_nodes)
            weight.randomizeGaussian();
            weight.multiply(Math.sqrt(2 / n_in));

            this.weights.push(weight);

            let bias = new Matrix(n_out, 1);
            // Biases can be initialized to 0 or slightly positive
            bias.randomizeGaussian();
            bias.multiply(0.1); // Small initialization
            this.biases.push(bias);
        }
    }

    // Forward pass
    feedforward(input_array) {
        if (input_array.length !== this.layer_sizes[0]) {
            console.error(`Input size ${input_array.length} does not match network input size ${this.layer_sizes[0]}`);
            return;
        }

        let current_activations = Matrix.fromArray(input_array);

        // This array will store activations for each layer (inputs, hidden1, hidden2, ..., output)
        // Note: We might not need to store all of them unless we need them for something else, 
        // but for training we usually re-calculate or return just the output. 
        // However, for backprop, we might want a different method or to store them.
        // For simplicity in this method, we just return the final output.
        // But for training, we'll need the intermediate values.

        for (let i = 0; i < this.weights.length; i++) {
            let weights = this.weights[i];
            let biases = this.biases[i];

            // Calculate Z = W * A + b
            current_activations = Matrix.multiply(weights, current_activations);
            current_activations.add(biases);

            // Apply activation function
            // Use hidden activation for all layers except the last one
            if (i === this.weights.length - 1) {
                current_activations.map(this.activation_output.func);
            } else {
                current_activations.map(this.activation_hidden.func);
            }
        }

        return current_activations.toArray();
    }

    train(input_array, target_array) {
        // --- Forward Pass ---
        let inputs = Matrix.fromArray(input_array);
        let targets = Matrix.fromArray(target_array);

        let layers_activations = [inputs];
        let current_activations = inputs;

        for (let i = 0; i < this.weights.length; i++) {
            let weights = this.weights[i];
            let biases = this.biases[i];

            // Linear transformation: z = W * a_prev + b
            let next_layer_z = Matrix.multiply(weights, current_activations);
            next_layer_z.add(biases);

            // Activation
            if (i === this.weights.length - 1) {
                // Output layer often uses Sigmoid
                next_layer_z.map(this.activation_output.func);
            } else {
                // Hidden layers use ReLU
                next_layer_z.map(this.activation_hidden.func);
            }

            current_activations = next_layer_z;
            layers_activations.push(current_activations);
        }

        let outputs = current_activations;

        // --- Backpropagation ---

        // 1. Calculate Output Error & Gradient
        let output_errors = Matrix.subtract(targets, outputs);

        // Gradient for Output Layer
        let gradients = Matrix.map(outputs, this.activation_output.dfunc);
        gradients.multiply(output_errors);
        gradients.multiply(this.learning_rate);

        // Calculate deltas for last weights
        let prev_layer_activations = layers_activations[layers_activations.length - 2];
        let prev_activations_T = Matrix.transpose(prev_layer_activations);
        let delta_weights_output = Matrix.multiply(gradients, prev_activations_T);

        // Update Output Layer Weights/Biases
        this.weights[this.weights.length - 1].add(delta_weights_output);
        this.biases[this.biases.length - 1].add(gradients);

        // 2. Propagate Error Backwards
        let next_layer_errors = output_errors; // Error at layer i+1 (initially Output)

        // Loop from last hidden layer back to first hidden layer
        for (let i = this.weights.length - 2; i >= 0; i--) {
            // Error propagation
            // error_l = weights_l+1_Transposed * error_l+1
            let weights_next = this.weights[i + 1];
            let weights_next_T = Matrix.transpose(weights_next);
            let hidden_errors = Matrix.multiply(weights_next_T, next_layer_errors);

            // Gradient for this hidden layer
            let current_layer_activations = layers_activations[i + 1];
            let hidden_gradients = Matrix.map(current_layer_activations, this.activation_hidden.dfunc);
            hidden_gradients.multiply(hidden_errors);
            hidden_gradients.multiply(this.learning_rate);

            // Deltas
            let prev_activations = layers_activations[i];
            let prev_T = Matrix.transpose(prev_activations);
            let delta_weights = Matrix.multiply(hidden_gradients, prev_T);

            // Update Weights/Biases
            this.weights[i].add(delta_weights);
            this.biases[i].add(hidden_gradients);

            // Set error for next iteration
            next_layer_errors = hidden_errors;
        }
    }
}
