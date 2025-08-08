// Fetch random items and render rating inputs
fetch('/api/random-items')
    .then(res => res.json())
    .then(data => {
        const container = document.getElementById('rating-form');
        data.items.forEach(pid => {
            const div = document.createElement('div');
            div.innerHTML = `Product ${pid}: <input type="number" min="1" max="5" id="rate-${pid}" />`;
            container.appendChild(div);
        });
    });

// On submit, gather ratings and request recommendations
document.getElementById('submit-btn').addEventListener('click', () => {
    const inputs = document.querySelectorAll('#rating-form input');
    const ratings = {};
    inputs.forEach(inp => {
        const val = inp.value;
        if (val) {
            const pid = inp.id.split('-')[1];
            ratings[pid] = Number(val);
        }
    });

    fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ratings })
    })
        .then(res => res.json())
        .then(data => {
            const list = document.getElementById('recommendations');
            list.innerHTML = '';
            data.recommendations.forEach(pid => {
                const li = document.createElement('li');
                li.textContent = `Product ${pid}`;
                list.appendChild(li);
            });
        });
});
