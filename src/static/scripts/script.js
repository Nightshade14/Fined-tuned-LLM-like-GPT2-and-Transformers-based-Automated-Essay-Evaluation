document.getElementById('evaluateButton').addEventListener('click', async function () {
    // Get the essay content and selected model
    const essay = document.getElementById("essay").value;
    const modelId = document.getElementById("model_id").value;

    const response = await fetch("http://127.0.0.1:8000/evaluate/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            essay: essay,
            my_model_id: modelId
        })
    });

    // Perform the evaluation (replace with your own logic)
    const score = await response.json();
    var feedback = 'Your essay has been evaluated.';

    // Display the result
    var resultDiv = document.getElementById('result');
    resultDiv.querySelector('.score').textContent = score.predicted_class + ' out of 5';
    resultDiv.querySelector('.feedback').textContent = feedback;
    resultDiv.style.display = 'block';
});