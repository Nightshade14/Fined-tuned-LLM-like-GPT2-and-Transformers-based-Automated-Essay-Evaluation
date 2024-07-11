document.getElementById("evaluateButton").addEventListener("click", async function () {
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

    const result = await response.json();
    console.log("Evaluation Result:" + result.predicted_class)
    document.getElementById("result").innerText = "Evaluation Result: " + (result.predicted_class || result.error) + "/5";
});