document.getElementById('fraudForm').addEventListener('submit', async function(e) {
    // Remove this block if you use classic Flask POST (not AJAX)
    if (window.location.pathname === "/") return; // Let Flask handle POST

    e.preventDefault();
    const resultDiv = document.getElementById('result');
    resultDiv.style.display = 'none';
    resultDiv.classList.remove('error');

    const data = {
        time_since_login_min: Number(document.getElementById('time_since_login_min').value),
        transaction_amount: Number(document.getElementById('transaction_amount').value),
        is_first_transaction: Number(document.getElementById('is_first_transaction').value),
        user_tenure_months: Number(document.getElementById('user_tenure_months').value),
        transaction_type: document.getElementById('transaction_type').value
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        const res = await response.json();
        if (response.ok) {
            resultDiv.textContent = `Prediction: ${res.is_fraud ? 'FRAUD' : 'NOT FRAUD'} (Probability: ${res.fraud_probability.toFixed(4)})`;
            resultDiv.style.display = 'block';
            resultDiv.classList.remove('error');
        } else {
            resultDiv.textContent = res.error || 'Prediction failed.';
            resultDiv.style.display = 'block';
            resultDiv.classList.add('error');
        }
    } catch (err) {
        resultDiv.textContent = 'Error connecting to prediction service.';
        resultDiv.style.display = 'block';
        resultDiv.classList.add('error');
    }
});
