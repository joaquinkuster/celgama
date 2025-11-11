document.addEventListener('DOMContentLoaded', function () {
    const $ = sel => document.querySelector(sel);
    const $$ = sel => document.querySelectorAll(sel);

    const introScreen = $('#introScreen');
    const formScreen = $('#formScreen');
    const resultScreen = $('#resultScreen');
    const startBtn = $('#startBtn');
    const phoneForm = $('#phoneForm');
    const progressBar = $('#progressBar');
    const progressSteps = $('#progressSteps');

    const steps = $$('.form-step');
    const totalSteps = steps.length;
    let currentStep = 0;
    let chartDifRel = null;
    let chartScatter = null;
    let chartDistribution = null;

    // ===== ANIMACIONES =====
    function animateTransition(hideEl, showEl) {
        anime({
            targets: hideEl,
            opacity: 0,
            translateY: -30,
            duration: 400,
            easing: 'easeInOutQuad',
            complete: () => {
                hideEl.classList.add('hidden');
                showEl.classList.remove('hidden');
                anime({
                    targets: showEl,
                    opacity: [0, 1],
                    translateY: [30, 0],
                    duration: 400,
                    easing: 'easeOutQuad'
                });
            }
        });
    }

    // ===== PROGRESO =====
    function initProgressSteps() {
        progressSteps.innerHTML = '';
        for (let i = 0; i < totalSteps; i++) {
            const step = document.createElement('div');
            step.className = 'step-indicator';
            step.innerHTML = `<div class="step-dot"></div><span>Paso ${i + 1}</span>`;
            progressSteps.appendChild(step);
        }
    }

    function updateProgress() {
        const progress = ((currentStep + 1) / totalSteps) * 100;
        progressBar.style.width = `${progress}%`;

        $$('.step-indicator').forEach((indicator, idx) => {
            indicator.classList.remove('active', 'completed');
            if (idx === currentStep) indicator.classList.add('active');
            else if (idx < currentStep) indicator.classList.add('completed');
        });
    }

    function showStep(stepIndex) {
        steps.forEach(step => step.classList.remove('active'));
        if (steps[stepIndex]) {
            steps[stepIndex].classList.add('active');
            currentStep = stepIndex;
            updateProgress();
        }
    }

    function validateStep() {
        const inputs = steps[currentStep].querySelectorAll('input[required]');
        for (let input of inputs) {
            if (!input.value || input.value.trim() === '') {
                input.classList.add('error');
                anime({ targets: input, translateX: [-5, 5, -5, 5, 0], duration: 200 });
                setTimeout(() => input.classList.remove('error'), 500);
                return false;
            }
        }
        return true;
    }

    // ===== NAVEGACIÓN =====
    initProgressSteps();
    showStep(0);

    startBtn.addEventListener('click', () => animateTransition(introScreen, formScreen));

    $$('.next-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            if (validateStep()) showStep(currentStep + 1);
        });
    });

    $$('.prev-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            if (currentStep > 0) showStep(currentStep - 1);
        });
    });

    // ===== ENVÍO =====
    phoneForm.addEventListener('submit', function (e) {
        e.preventDefault();
        if (!validateStep()) return;

        const submitBtn = $('#submitBtn');
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Procesando...';
        submitBtn.disabled = true;

        const formData = new FormData(phoneForm);
        const data = {};
        for (let [key, value] of formData.entries()) {
            data[key] = value;
        }
        if (!data.fast_charging_available) data.fast_charging_available = '0';

        fetch('/api/resultado', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        })
            .then(res => res.ok ? res.json() : res.json().then(err => Promise.reject(err)))
            .then(resultado => mostrarResultado(resultado))
            .catch(error => alert(`Error: ${error.error || error.message}`))
            .finally(() => {
                submitBtn.innerHTML = '<i class="fas fa-chart-bar"></i> Ver resultado';
                submitBtn.disabled = false;
            });
    });

    // ===== MOSTRAR RESULTADOS =====
    function mostrarResultado(resultado) {
        // Título y datos básicos
        $('#resultGama').textContent = resultado.gama;
        $('#explanationGama').textContent = resultado.gama.replace('Gama ', '');
        $('#totalDispositivos').textContent = resultado.total_dispositivos;
        $('#keyFactors').textContent = resultado.factores_clave.join(', ');

        // Lista de características
        const featuresList = $('#featuresList');
        featuresList.innerHTML = '';
        for (const [key, value] of Object.entries(resultado.promedio)) {
            const li = document.createElement('li');
            li.innerHTML = `<span>${key}</span><span>${value}</span>`;
            featuresList.appendChild(li);
        }

        // Gráfico de diferencias relativas (Barras)
        if (chartDifRel) chartDifRel.destroy();
        const ctxDiff = $('#chartDifRel').getContext('2d');
        chartDifRel = new Chart(ctxDiff, {
            type: 'bar',
            data: {
                labels: Object.keys(resultado.dif_relativas),
                datasets: [{
                    label: 'Diferencia relativa (%)',
                    data: Object.values(resultado.dif_relativas).map(v => v * 100),
                    backgroundColor: Object.values(resultado.dif_relativas).map(v =>
                        v >= 0 ? 'rgba(0, 150, 0, 0.7)' : 'rgba(200, 0, 0, 0.7)'
                    ),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: ctx => `${ctx.parsed.y.toFixed(1)}%`
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Diferencia (%)' }
                    }
                }
            }
        });

        // Gráfico de dispersión (PCA) 
        if (chartScatter) chartScatter.destroy();
        const ctxScatter = $('#chartScatter').getContext('2d');

        console.log(resultado.pca_usuario)

        chartScatter = new Chart(ctxScatter, {
            type: 'scatter',
            data: {
                datasets: [
                    // Regiones suaves (una por gama)
                    {
                        label: 'Regiones de Gama',
                        data: resultado.pca_clusters.map((p, i) => ({ x: p[0], y: p[1] })),
                        backgroundColor: [
                            'rgba(76,175,80,0.1)',   // Alta
                            'rgba(255,193,7,0.1)',   // Media
                            'rgba(244,67,54,0.1)'    // Baja
                        ],
                        pointRadius: 100,
                        borderWidth: 0,
                        showLine: false,
                        order: 0,
                        pointHoverRadius: 0,   // 🔹 no cambia tamaño al hover
                        pointHitRadius: 0,     // 🔹 no capta el mouse
                        hoverBackgroundColor: 'transparent', // 🔹 evita efecto hover
                    },
                    // Centroides (clusters)
                    {
                        label: 'Centroides de Gamas',
                        data: resultado.pca_clusters.map((p, i) => ({
                            x: p[0],
                            y: p[1],
                            gama: resultado.gamas[i]
                        })),
                        backgroundColor: resultado.gamas.map(g =>
                            g === 'Alta' ? 'rgba(76,175,80,0.8)' :
                                g === 'Media' ? 'rgba(255,193,7,0.8)' :
                                    'rgba(244,67,54,0.8)'
                        ),
                        pointRadius: 10,
                        //borderWidth: 2,
                        //borderColor: '#000',
                        order: 1
                    },
                    // Punto del usuario
                    {
                        label: 'Tu dispositivo',
                        data: [{ x: resultado.pca_usuario[0], y: resultado.pca_usuario[1] }],
                        backgroundColor: 'rgba(33,150,243,1)',
                        //pointStyle: 'star',
                        //borderColor: '#000',
                        //borderWidth: 2,
                        pointRadius: 5,
                        order: 999 // 🔹 Siempre arriba de todo
                    }
                ]
            },
            options: {
                plugins: {
                    datalabels: {
                        align: 'top',
                        formatter: (v) => v.gama || '',
                        color: '#000',
                        font: { weight: 'bold', size: 14 }
                    },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => {
                                if (ctx.dataset.label === 'Tu dispositivo') {
                                    return `Tu dispositivo (más cercano: ${resultado.gama_cercana}, dist=${resultado.distancia_minima})`;
                                } else if (ctx.raw.gama) {
                                    return `${ctx.raw.gama}`;
                                } else {
                                    return '';
                                }
                            }
                        }
                    },
                    legend: { position: 'top' }
                },
                scales: {
                    x: { title: { display: true, text: 'Componente 1 (PCA)' } },
                    y: { title: { display: true, text: 'Componente 2 (PCA)' } }
                }
            },
            plugins: [ChartDataLabels]
        });

        // Gráfico de distribución (Circular)
        if (chartDistribution) chartDistribution.destroy();
        const ctxDist = $('#chartDistribution').getContext('2d');
        const gamas = Object.keys(resultado.distribucion);
        const cantidades = Object.values(resultado.distribucion);
        const colores = ['#ff6b6b', '#ffd93d', '#6bcf7f'];

        chartDistribution = new Chart(ctxDist, {
            type: 'pie',
            data: {
                labels: gamas,
                datasets: [{
                    data: cantidades,
                    backgroundColor: colores,
                    borderColor: '#fff',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'bottom' },
                    tooltip: {
                        callbacks: {
                            label: ctx => {
                                const total = cantidades.reduce((a, b) => a + b, 0);
                                const porcentaje = ((ctx.parsed / total) * 100).toFixed(1);
                                return `${ctx.label}: ${ctx.parsed} (${porcentaje}%)`;
                            }
                        }
                    }
                }
            }
        });

        animateTransition(formScreen, resultScreen);
    }

    // ===== REINICIAR =====
    $('#restartBtn').addEventListener('click', () => {
        animateTransition(resultScreen, formScreen);
        phoneForm.reset();
        currentStep = 0;
        showStep(0);
    });
});