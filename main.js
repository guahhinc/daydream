const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const resolution = 5; // Higher res for smoother spiral
const cols = canvas.width / resolution;
const rows = canvas.height / resolution;

let training_data = [];
let nn;

// --- Math Spiral Generator ---
function generateSpiralData(pointsPerArm) {
    let data = [];

    // Iterate for 2 arms (Red and Cyan)
    // Arm 0: Red [1, 0, 0]
    // Arm 1: Cyan [0, 1, 1]

    for (let arm = 0; arm < 2; arm++) {
        for (let i = 0; i < pointsPerArm; i++) {
            // Radius: 0 to 1
            let r = i / pointsPerArm;

            // Angle: Rotates 2.5 times (5*PI) plus phase shift for each arm
            let theta = 2.5 * Math.PI * r + (arm * Math.PI);

            // Add random noise
            let noiseX = (Math.random() - 0.5) * 0.05;
            let noiseY = (Math.random() - 0.5) * 0.05;

            // Polar to Cartesian
            let x = (r * Math.cos(theta)) + noiseX;
            let y = (r * Math.sin(theta)) + noiseY;

            // Normalize from [-1, 1] to [0, 1] for the Neural Network
            let inputX = (x + 1) / 2;
            let inputY = (y + 1) / 2;

            let target;
            if (arm === 0) target = [1, 0, 0]; // Red
            else target = [0, 1, 1]; // Cyan

            data.push({ inputs: [inputX, inputY], targets: target });
        }
    }
    return data;
}

function resetNetwork() {
    // Gigantic Neural Network
    // 2 Inputs -> 16 -> 16 -> 16 -> 3 Outputs (RGB)
    nn = new NeuralNetwork([2, 16, 16, 16, 3]);
    nn.learning_rate = 0.03; // Slightly lower for complex shapes

    // Generate fresh data
    training_data = generateSpiralData(500); // 500 points per arm
}

// Fisher-Yates Shuffle
function shuffle(array) {
    let currentIndex = array.length, randomIndex;
    while (currentIndex != 0) {
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex--;
        [array[currentIndex], array[randomIndex]] = [
            array[randomIndex], array[currentIndex]];
    }
    return array;
}

function draw() {
    // Train the network!
    // Train multiple batches per frame
    for (let k = 0; k < 10; k++) { // Reduced batches per frame to keep UI smooth with more data
        let training_batch = shuffle([...training_data]);
        for (let i = 0; i < training_batch.length; i++) {
            nn.train(training_batch[i].inputs, training_batch[i].targets);
        }
    }

    // Visualization: The Decision Boundary (Background)

    // Optimization: Get ImageData to write pixels directly (much faster than fillRect)
    // But for simplicity and readability, keeping the loop, but skipping pixels could help
    // For now, let's stick to fillRect but maybe lower resolution if slow.

    for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
            let x1 = i / cols;
            let x2 = j / rows;

            let inputs = [x1, x2];
            let output = nn.feedforward(inputs);

            let r = output[0] * 255;
            let g = output[1] * 255;
            let b = output[2] * 255;

            ctx.fillStyle = `rgb(${r},${g},${b})`;
            ctx.fillRect(i * resolution, j * resolution, resolution, resolution);
        }
    }

    // Draw Training Points overlay
    for (let pt of training_data) {
        // Map back to canvas coords
        let px = pt.inputs[0] * canvas.width;
        let py = pt.inputs[1] * canvas.height;

        ctx.beginPath();
        ctx.fillStyle = pt.targets[0] === 1 ? '#ff0000' : '#00ffff';
        ctx.arc(px, py, 2, 0, Math.PI * 2);
        ctx.fill();
    }

    requestAnimationFrame(draw);
}

// Start
resetNetwork();
draw();
