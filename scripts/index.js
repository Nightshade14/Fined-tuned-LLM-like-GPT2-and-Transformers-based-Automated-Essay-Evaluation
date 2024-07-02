document.getElementById("evaluateButton").addEventListener("click", async function () {
    const essay = document.getElementById("essay").value;
    const modelId = document.getElementById("model_id").value;

    console.log("Essay:", essay);
    console.log("Model ID:", modelId);

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

    const result = await response.json();
    document.getElementById("result").innerText = "Evaluation Result: " + (result.my_model_id || result.error);
});