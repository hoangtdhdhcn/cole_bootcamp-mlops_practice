document.getElementById('predictionForm').addEventListener('submit', async (event) => {
    event.preventDefault();

    // Get input values and ensure they are numbers
    const petalLength = parseFloat(document.getElementById('petal_length').value);
    const petalWidth = parseFloat(document.getElementById('petal_width').value);

    // Check if the inputs are valid numbers
    if (isNaN(petalLength) || isNaN(petalWidth)) {
        alert('Please enter valid numeric values for both Petal Length and Petal Width.');
        return;
    }

    // Send data to the FastAPI backend
    const response = await fetch('/submit', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
            petal_length: petalLength,
            petal_width: petalWidth
        })
    });

    // Handle the response from the server
    const result = await response.json();

    // Check if the species prediction is available
    if (result.species) {
        document.getElementById('species').textContent = `Predicted Species: ${result.species}`;
    } else {
        document.getElementById('species').textContent = 'Error: Could not predict the species.';
    }
});
