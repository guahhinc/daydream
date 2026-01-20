const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const resolution = 10;
const cols = canvas.width / resolution;
const rows = canvas.height / resolution;

// Colorful "XOR" Problem Training Data
// Mapping corners to specific colors
const training_data = [
    { inputs: [0, 0], targets: [1, 0, 0] }, // Red
    { inputs: [0, 1], targets: [0, 1, 0] }, // Green
    { inputs: [1, 0], targets: [0, 0, 1] }, // Blue
    { inputs: [1, 1], targets: [0, 1, 1] }  // Cyan
];

let nn;

function resetNetwork() {
    // Gigantic Neural Network
    // 2 Inputs -> 16 -> 16 -> 16 -> 3 Outputs (RGB)
    nn = new NeuralNetwork([2, 16, 16, 16, 3]);
    nn.learning_rate = 0.05; // Lower LR for stability with deep networks
}

// Fisher-Yates Shuffle
function shuffle(array) {
    let currentIndex = array.length, randomIndex;

    // While there remain elements to shuffle.
    while (currentIndex != 0) {

        // Pick a remaining element.
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex--;

        // And swap it with the current element.
        [array[currentIndex], array[randomIndex]] = [
            array[randomIndex], array[currentIndex]];
    }

    return array;
}

function draw() {
    // Train the network!
    // Stochastic Gradient Descent with Shuffling

    // Train multiple batches per frame
    for (let k = 0; k < 50; k++) {
        let training_batch = shuffle([...training_data]); // Copy and shuffle
        for (let i = 0; i < training_batch.length; i++) {
            nn.train(training_batch[i].inputs, training_batch[i].targets);
        }
    }

    // Visualization
    // Loop through every 'pixel' (resolution block)
    for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
            // Normalize coordinates to 0-1
            let x1 = i / cols;
            let x2 = j / rows;

            // Ask the network what it thinks this pixel should be
            let inputs = [x1, x2];
            let output = nn.feedforward(inputs);

            // Draw
            // Output is [r, g, b] between 0 and 1.
            let r = output[0] * 255;
            let g = output[1] * 255;
            let b = output[2] * 255;

            ctx.fillStyle = `rgb(${r},${g},${b})`;
            ctx.fillRect(i * resolution, j * resolution, resolution, resolution);
        }
    }

    requestAnimationFrame(draw);
}

// Start
resetNetwork();
draw();
