const form = document.getElementById('predict-form');
const resultDiv = document.getElementById('result');

form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const symptomsInput = document.getElementById('symptoms').value;
    const symptoms = symptomsInput.split(',').map(s => s.trim());

    resultDiv.innerHTML = "⏳ Loading...";

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({symptoms: symptoms})
        });

        const data = await response.json();

        if (data.results) {
            let output = "<h2>Results:</h2>";
            data.results.forEach(d => {
                output += `
                    <div>
                        <h3>${d.Disease}</h3>
                        <p><strong>Matched Symptoms:</strong> ${d.Matched_Symptoms.join(", ")}</p>
                        <p><strong>Score:</strong> ${d.Score}</p>
                        <p><strong>Description:</strong> ${d.Description}</p>
                        <p><strong>Recommended Drugs:</strong> ${d.Recommended_Drugs}</p>
                        <p><strong>Test Suggestions:</strong> ${d.Test_Suggestions}</p>
                        <p><strong>Specialist:</strong> ${d.Specialist}</p>
                        <hr>
                    </div>
                `;
            });
            resultDiv.innerHTML = output;
        } else {
            resultDiv.innerHTML = `<p style="color:red;">${data.error || "No results found"}</p>`;
        }
    } catch (error) {
        resultDiv.innerHTML = `<p style="color:red;">Error: ${error}</p>`;
    }
});
