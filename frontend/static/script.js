document.addEventListener('DOMContentLoaded', () => {
    const newsTextarea = document.getElementById('newsText');
    const charCountEl = document.getElementById('char-count');
    const statusBanner = document.getElementById('status-banner');
    const statusText = document.getElementById('status-text');
    const exampleBadges = document.querySelectorAll('.example-badge');
    const predictionForm = document.getElementById('prediction-form');

    // 1. Real-time Character Counter
    newsTextarea.addEventListener('input', () => {
        const count = newsTextarea.value.length;
        charCountEl.textContent = `${count} character${count !== 1 ? 's' : ''}`;
    });

    // 2. Fetch Backend Status on Load
    fetch('/status')
        .then(response => response.json())
        .then(data => {
            statusBanner.classList.remove('d-none');
            if (data.model_loaded) {
                statusBanner.className = 'alert-custom alert-success-custom mb-4 fade-in';
                statusText.innerHTML = '<strong>System Online:</strong> Google MuRIL Classifier & RAG Fact-Check Engine are fully loaded.';
                // Hide status banner after 4 seconds
                setTimeout(() => {
                    statusBanner.style.transition = 'opacity 0.8s ease';
                    statusBanner.style.opacity = '0';
                    setTimeout(() => statusBanner.classList.add('d-none'), 800);
                }, 4000);
            } else {
                statusBanner.className = 'alert-custom alert-warning-custom mb-4 fade-in';
                statusText.innerHTML = '<strong>Engine Loading / Training:</strong> The transformer model is loading or running training. Predictions may temporarily use a fallback model.';
            }
        })
        .catch(err => {
            console.error("Error fetching system status:", err);
            statusBanner.classList.remove('d-none');
            statusBanner.className = 'alert-custom alert-warning-custom mb-4 fade-in';
            statusText.innerHTML = '<strong>Offline Mode:</strong> Failed to reach backend status server. Checking prediction capabilities...';
        });

    // 3. Quick Examples Click Handlers
    exampleBadges.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const text = btn.getAttribute('data-text');
            newsTextarea.value = text;
            
            // Trigger character counter update
            const event = new Event('input', { bubbles: true });
            newsTextarea.dispatchEvent(event);

            // Scroll textarea into view on small screens
            newsTextarea.scrollIntoView({ behavior: 'smooth', block: 'center' });
        });
    });

    // 4. Form Submission Handler
    predictionForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const text = newsTextarea.value.trim();
        if (!text) return;

        const predictBtn = document.getElementById('analyze-btn');
        const loadingSpinner = document.getElementById('btn-spinner');
        const btnText = document.getElementById('btn-text');
        
        const placeholderSection = document.getElementById('results-placeholder');
        const loadingSection = document.getElementById('results-loading');
        const resultSection = document.getElementById('result-section');
        
        const predictionBadge = document.getElementById('prediction-badge');
        const confidenceText = document.getElementById('confidence-text');
        const confidenceBar = document.getElementById('confidence-bar');
        const reasoningBox = document.getElementById('reasoning-box');
        const evidenceList = document.getElementById('evidence-list');
        
        // UI Loading States
        predictBtn.disabled = true;
        loadingSpinner.classList.remove('d-none');
        btnText.textContent = "Running Verification...";
        
        placeholderSection.classList.add('d-none');
        resultSection.classList.add('d-none');
        loadingSection.classList.remove('d-none');
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                const isReal = data.result === 'real';
                const confidencePercent = Math.round(data.confidence * 100);
                
                // 1. Set Classification Output & Badges
                predictionBadge.className = `result-badge ${isReal ? 'result-real' : 'result-fake'}`;
                predictionBadge.innerHTML = isReal ? 
                    '<i class="bi bi-shield-check fs-4"></i> LIKELY REAL NEWS' : 
                    '<i class="bi bi-shield-x fs-4"></i> LIKELY FAKE NEWS';
                
                // 2. Set Confidence Progress Bar
                confidenceText.textContent = `${confidencePercent}%`;
                confidenceBar.className = `progress-bar progress-bar-striped progress-bar-animated custom-progress-bar ${isReal ? 'progress-bar-real' : 'progress-bar-fake'}`;
                confidenceBar.style.width = `${confidencePercent}%`;
                
                // 3. Set RAG Reasoning Explanation
                reasoningBox.innerHTML = `<strong>AI Explainer & Context:</strong><br>${data.reasoning || 'No analysis explanation generated.'}`;
                
                // 4. Render RAG Live Web Evidence
                evidenceList.innerHTML = '';
                if (data.evidence && data.evidence.length > 0) {
                    data.evidence.forEach(item => {
                        const similarityPercent = Math.round(item.similarity * 100);
                        let badgeClass = 'bg-danger';
                        if (item.similarity > 0.75) {
                            badgeClass = 'bg-success';
                        } else if (item.similarity > 0.5) {
                            badgeClass = 'bg-warning text-dark';
                        }
                        
                        const el = document.createElement('a');
                        el.href = item.link || '#';
                        el.target = "_blank";
                        el.className = "list-group-item list-group-item-action d-flex justify-content-between align-items-center py-3 evidence-item";
                        el.innerHTML = `
                            <div>
                                <h6 class="mb-1 text-info fw-semibold" style="font-size: 0.95rem;">${item.title}</h6>
                                <small class="text-secondary">Publisher: <span class="badge bg-dark border border-secondary px-2 py-1 ms-1">${item.source}</span></small>
                            </div>
                            <span class="badge ${badgeClass} rounded-pill px-3 py-2 fw-bold ms-2">
                                ${similarityPercent}% Match
                            </span>
                        `;
                        evidenceList.appendChild(el);
                    });
                } else {
                    evidenceList.innerHTML = `
                        <div class="text-secondary fst-italic p-3 text-center bg-dark bg-opacity-20 rounded-3 border border-secondary border-opacity-10">
                            <i class="bi bi-search-heart me-1 text-muted"></i> No corroborating live fact-checks found on global news feeds.
                        </div>`;
                }
                
                // Transition UI States
                loadingSection.classList.add('d-none');
                resultSection.classList.remove('d-none');
                
            } else {
                alert('Verification Error: ' + (data.error || 'The model was unable to process the request.'));
                loadingSection.classList.add('d-none');
                placeholderSection.classList.remove('d-none');
            }
        } catch (error) {
            console.error('Connection Error:', error);
            alert('Failed to connect to the backend server. Make sure the API is active.');
            loadingSection.classList.add('d-none');
            placeholderSection.classList.remove('d-none');
        } finally {
            // Restore button state
            predictBtn.disabled = false;
            loadingSpinner.classList.add('d-none');
            btnText.textContent = "Verify Authenticity";
        }
    });
});
