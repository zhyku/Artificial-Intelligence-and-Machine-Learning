document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("prediction-form");
    const resultEl = document.getElementById("result");

    if (!form) {
        console.error("Form element not found");
        return;
    }

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const formData = collectFormData();

        if (!validateFormData(formData)) {
            alert("Please provide valid input values.");
            return;
        }

        try {
            const result = await fetchPrediction(formData);
            displayResult(result);
        } catch (err) {
            console.error("Error during fetch:", err);
            resultEl.textContent = "Error: Unable to connect to server.";
            resultEl.className = "text-danger";
        }
    });
});

function collectFormData() {
    return {
        hour: parseInt(getElementValue("hour")) || 0,
        temperature: parseFloat(getElementValue("temperature")) || 0.0,
        humidity: parseInt(getElementValue("humidity")) || 0,
        wind_speed: parseFloat(getElementValue("wind_speed")) || 0.0,
        visibility: parseInt(getElementValue("visibility")) || 0,
        dew_point: parseFloat(getElementValue("dew_point")) || 0.0,
        solar_radiation: parseFloat(getElementValue("solar_radiation")) || 0.0,
        rainfall: parseFloat(getElementValue("rainfall")) || 0.0,
        snowfall: parseFloat(getElementValue("snowfall")) || 0.0,
        seasons: getElementValue("seasons") || "Unknown",
        holiday: getElementValue("holiday") || "No",
        functioning_day: getElementValue("functioning_day") || "Yes"
    };
}

function getElementValue(id) {
    const element = document.getElementById(id);
    return element ? element.value.trim() : null;
}

function validateFormData(data) {
    for (const key in data) {
        if (data[key] === null || data[key] === "" || (typeof data[key] !== "string" && isNaN(data[key]))) {
            alert(`Please enter a valid value for ${key.replace("_", " ")}`);
            return false;
        }
    }
    return true;
}

async function fetchPrediction(formData) {
    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData)
    });

    return await response.json();
}

function displayResult(result) {
    const resultEl = document.getElementById("result");

    if (!resultEl) {
        console.error("Result element not found");
        return;
    }

    if (result.error) {
        resultEl.textContent = `Error: ${result.error}`;
        resultEl.className = "text-danger";
    } else {
        resultEl.textContent = `Predicted Bike Rentals: ${Math.round(result.prediction)}`;
        resultEl.className = "text-success";
    }
}
