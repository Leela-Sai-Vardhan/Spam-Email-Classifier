document.addEventListener('DOMContentLoaded', () => {
    loadExamples();
    loadModelComparison();
    loadTopFeatures();
    createParticles();
});

function createParticles() {
    const container = document.getElementById('particles');
    for (let i = 0; i < 30; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 15 + 's';
        particle.style.animationDuration = (10 + Math.random() * 20) + 's';
        container.appendChild(particle);
    }
}

async function loadExamples() {
    try {
        const res = await fetch('/api/examples');
        const examples = await res.json();
        const container = document.getElementById('exampleTags');
        container.innerHTML = examples.map(ex =>
            `<div class="example-tag ${ex.type}" onclick="useExample('${escapeHtml(ex.text).replace(/'/g, "\\'")}')">${escapeHtml(ex.text.substring(0, 40))}${ex.text.length > 40 ? '...' : ''}</div>`
        ).join('');
    } catch (err) {
        console.error('Failed to load examples:', err);
    }
}

async function loadModelComparison() {
    try {
        const res = await fetch('/api/model-comparison');
        const data = await res.json();
        const container = document.getElementById('modelComparison');
        const colors = ['#4a9eff', '#bb86fc', '#00e676'];
        let i = 0;
        container.innerHTML = Object.entries(data).map(([key, val]) =>
            `<div class="model-item" style="border-left-color: ${colors[i++ % 3]}">
                <span class="model-name">${val.name}</span>
                <span class="model-accuracy">${val.accuracy}%</span>
            </div>`
        ).join('');
    } catch (err) {
        console.error('Failed to load model comparison:', err);
    }
}

async function loadTopFeatures() {
    try {
        const res = await fetch('/api/top-features');
        const data = await res.json();
        renderFeatures('spamFeatures', data.spam_words || []);
        renderFeatures('hamFeatures', data.ham_words || []);
    } catch (err) {
        console.error('Failed to load features:', err);
    }
}

function renderFeatures(containerId, features) {
    const container = document.getElementById(containerId);
    container.innerHTML = features.slice(0, 8).map(f =>
        `<div class="feature-item">
            <span class="feature-word">${f.word}</span>
            <span class="feature-score">${f.score.toFixed(3)}</span>
        </div>`
    ).join('');
}

function useExample(text) {
    document.getElementById('messageInput').value = text;
    classifyMessage();
}

async function classifyMessage() {
    const input = document.getElementById('messageInput').value.trim();
    if (!input) {
        alert('Please enter a message to classify.');
        return;
    }

    const btn = document.getElementById('classifyBtn');
    btn.innerHTML = '<span class="loading"></span> Analyzing...';
    btn.disabled = true;

    try {
        const res = await fetch('/api/classify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: input })
        });
        const data = await res.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        showResult(data);
    } catch (err) {
        alert('Error connecting to server. Make sure the Flask app is running.');
    } finally {
        btn.innerHTML = '<span class="btn-icon">🔍</span> Analyze Message';
        btn.disabled = false;
    }
}

function showResult(data) {
    const card = document.getElementById('resultCard');
    const verdict = document.getElementById('verdict');
    const fill = document.getElementById('confidenceFill');
    const value = document.getElementById('confidenceValue');

    card.style.display = 'block';
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    if (data.is_spam) {
        verdict.className = 'verdict spam';
        verdict.innerHTML = '🚫 SPAM DETECTED';
        fill.className = 'confidence-fill high';
    } else {
        verdict.className = 'verdict ham';
        verdict.innerHTML = '✅ SAFE (Not Spam)';
        fill.className = 'confidence-fill';
    }

    fill.style.width = '0%';
    value.textContent = '0%';

    setTimeout(() => {
        fill.style.width = data.confidence + '%';
        animateCounter(value, 0, data.confidence, 1000);
    }, 100);
}

function animateCounter(element, start, end, duration) {
    const startTime = performance.now();
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = start + (end - start) * eased;
        element.textContent = current.toFixed(1) + '%';
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

function clearAll() {
    document.getElementById('messageInput').value = '';
    document.getElementById('resultCard').style.display = 'none';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
