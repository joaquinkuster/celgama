document.addEventListener('DOMContentLoaded', function () {
    // ===== SELECTORES =====
    const $ = sel => document.querySelector(sel);
    const $$ = sel => document.querySelectorAll(sel);

    const pantallaIntro = $('#introScreen');
    const pantallaFormulario = $('#formScreen');
    const pantallaResultado = $('#resultScreen');
    const botonIniciar = $('#startBtn');
    const formularioCelular = $('#phoneForm');
    const barraProgreso = $('#progressBar');
    const contenedorPasos = $('#progressSteps');

    const pasos = $$('.form-step');
    const totalPasos = pasos.length;
    let pasoActual = 0;
    
    // Variables para los gráficos
    let graficoBarras = null;
    let graficoDispersion = null;
    let graficoDistribucion = null;

    // ===== ANIMACIONES DE TRANSICIÓN =====
    function animarTransicion(elementoOcultar, elementoMostrar) {
        anime({
            targets: elementoOcultar,
            opacity: 0,
            translateY: -30,
            duration: 400,
            easing: 'easeInOutQuad',
            complete: () => {
                elementoOcultar.classList.add('hidden');
                elementoMostrar.classList.remove('hidden');
                anime({
                    targets: elementoMostrar,
                    opacity: [0, 1],
                    translateY: [30, 0],
                    duration: 400,
                    easing: 'easeOutQuad'
                });
            }
        });
    }

    // ===== SISTEMA DE PROGRESO =====
    function inicializarPasos() {
        contenedorPasos.innerHTML = '';
        for (let i = 0; i < totalPasos; i++) {
            const paso = document.createElement('div');
            paso.className = 'step-indicator';
            paso.innerHTML = `<div class="step-dot"></div><span>Paso ${i + 1}</span>`;
            contenedorPasos.appendChild(paso);
        }
    }

    function actualizarProgreso() {
        const progreso = ((pasoActual + 1) / totalPasos) * 100;
        barraProgreso.style.width = `${progreso}%`;

        $$('.step-indicator').forEach((indicador, indice) => {
            indicador.classList.remove('active', 'completed');
            if (indice === pasoActual) indicador.classList.add('active');
            else if (indice < pasoActual) indicador.classList.add('completed');
        });
    }

    function mostrarPaso(indicePaso) {
        pasos.forEach(paso => paso.classList.remove('active'));
        if (pasos[indicePaso]) {
            pasos[indicePaso].classList.add('active');
            pasoActual = indicePaso;
            actualizarProgreso();
            
            // Enfocar el primer input del paso
            const primerInput = pasos[indicePaso].querySelector('input');
            if (primerInput) primerInput.focus();
        }
    }

    // ===== VALIDACIÓN DE CAMPOS =====
    function validarPaso() {
        const inputs = pasos[pasoActual].querySelectorAll('input[required]');
        let esValido = true;

        for (let input of inputs) {
            // Validar si está vacío
            if (!input.value || input.value.trim() === '') {
                mostrarError(input);
                esValido = false;
                continue;
            }

            // Validar rango numérico
            if (input.type === 'number') {
                const valor = parseFloat(input.value);
                const minimo = parseFloat(input.min);
                const maximo = parseFloat(input.max);

                if (minimo && valor < minimo) {
                    mostrarError(input, `Valor mínimo: ${minimo}`);
                    esValido = false;
                    continue;
                }

                if (maximo && valor > maximo) {
                    mostrarError(input, `Valor máximo: ${maximo}`);
                    esValido = false;
                    continue;
                }
            }

            // Remover error si es válido
            input.classList.remove('error');
        }

        return esValido;
    }

    function mostrarError(input, mensaje = '') {
        input.classList.add('error');
        anime({ 
            targets: input, 
            translateX: [-5, 5, -5, 5, 0], 
            duration: 200 
        });
        
        if (mensaje) {
            // Mostrar tooltip con el error
            const tooltip = document.createElement('div');
            tooltip.className = 'error-tooltip';
            tooltip.textContent = mensaje;
            input.parentElement.appendChild(tooltip);
            setTimeout(() => tooltip.remove(), 3000);
        }
        
        setTimeout(() => input.classList.remove('error'), 500);
    }

    // ===== NAVEGACIÓN DEL FORMULARIO =====
    inicializarPasos();
    mostrarPaso(0);

    botonIniciar.addEventListener('click', () => {
        animarTransicion(pantallaIntro, pantallaFormulario);
    });

    // Botones "Siguiente"
    $$('.next-btn').forEach(boton => {
        boton.addEventListener('click', () => {
            if (validarPaso() && pasoActual < totalPasos - 1) {
                mostrarPaso(pasoActual + 1);
            }
        });
    });

    // Botones "Anterior"
    $$('.prev-btn').forEach(boton => {
        boton.addEventListener('click', () => {
            if (pasoActual > 0) mostrarPaso(pasoActual - 1);
        });
    });

    // Navegar con Enter
    formularioCelular.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            
            if (pasoActual < totalPasos - 1) {
                // Siguiente paso
                if (validarPaso()) mostrarPaso(pasoActual + 1);
            } else {
                // Último paso: enviar formulario
                if (validarPaso()) formularioCelular.requestSubmit();
            }
        }
    });

    // ===== ENVÍO DEL FORMULARIO =====
    formularioCelular.addEventListener('submit', function (e) {
        e.preventDefault();
        if (!validarPaso()) return;

        const botonEnviar = $('#submitBtn');
        botonEnviar.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Procesando...';
        botonEnviar.disabled = true;

        // Recopilar datos del formulario
        const datosFormulario = new FormData(formularioCelular);
        const datos = {};
        for (let [clave, valor] of datosFormulario.entries()) {
            datos[clave] = valor;
        }
        
        // Asegurar que fast_charging tenga un valor
        if (!datos.fast_charging_available) {
            datos.fast_charging_available = '0';
        }

        // Enviar a la API
        fetch('/api/resultado', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(datos)
        })
            .then(respuesta => respuesta.ok ? respuesta.json() : respuesta.json().then(err => Promise.reject(err)))
            .then(resultado => mostrarResultado(resultado, datos))
            .catch(error => {
                alert(`Error: ${error.error || error.message}`);
                console.error('Error completo:', error);
            })
            .finally(() => {
                botonEnviar.innerHTML = '<i class="fas fa-chart-bar"></i> Ver resultado';
                botonEnviar.disabled = false;
            });
    });

    // ===== MOSTRAR RESULTADOS =====
    function mostrarResultado(resultado, datosUsuario) {
        // Título y gama
        $('#resultGama').textContent = resultado.gama;
        $('#explanationGama').textContent = resultado.gama.replace('Gama ', '');
        $('#totalDispositivos').textContent = resultado.total_dispositivos;

        // Mostrar advertencia si está en zona ambigua
        if (resultado.es_ambiguo) {
            mostrarAdvertenciaAmbiguedad(resultado);
        }

        // Construir explicación extendida con diferencias
        const factoresClave = resultado.factores_clave.slice(0, 3);
        const diferenciasTexto = construirExplicacionDiferencias(resultado.dif_relativas, factoresClave);
        $('#keyFactors').innerHTML = diferenciasTexto;

        // Mostrar datos del usuario junto con promedios
        mostrarComparacionCompleta(datosUsuario, resultado.promedio);

        // Lista de características promedio
        const listaCaracteristicas = $('#featuresList');
        listaCaracteristicas.innerHTML = '';
        for (const [clave, valor] of Object.entries(resultado.promedio)) {
            const li = document.createElement('li');
            li.innerHTML = `<span>${clave}</span><span>${valor}</span>`;
            listaCaracteristicas.appendChild(li);
        }

        // Crear gráficos
        crearGraficoBarras(resultado.dif_relativas);
        crearGraficoDispersion(resultado);
        crearGraficoDistribucion(resultado.distribucion);

        // Mostrar pantalla de resultados
        animarTransicion(pantallaFormulario, pantallaResultado);
    }

    // ===== EXPLICACIÓN DETALLADA DE DIFERENCIAS =====
    function construirExplicacionDiferencias(diferenciasRelativas, factores) {
        const explicaciones = [];
        
        factores.forEach(factor => {
            // Encontrar la clave correspondiente en dif_relativas
            let categoriaEncontrada = '';
            let diferenciaValor = 0;
            
            // Mapeo de nombres legibles a categorías
            const mapeoFactores = {
                'precio': 'Precio',
                'memoria RAM': 'Memoria',
                'cámara principal': 'Cámara',
                'velocidad del procesador': 'Procesador',
                'capacidad de batería': 'Batería',
                'almacenamiento interno': 'Memoria'
            };
            
            categoriaEncontrada = mapeoFactores[factor] || '';
            
            if (categoriaEncontrada && diferenciasRelativas[categoriaEncontrada] !== undefined) {
                diferenciaValor = diferenciasRelativas[categoriaEncontrada] * 100;
                
                let interpretacion = '';
                const diferenciaAbs = Math.abs(diferenciaValor);
                
                if (diferenciaValor > 0) {
                    interpretacion = `<span class="diferencia-positiva">+${diferenciaAbs.toFixed(1)}% superior</span>`;
                } else {
                    interpretacion = `<span class="diferencia-negativa">${diferenciaAbs.toFixed(1)}% inferior</span>`;
                }
                
                explicaciones.push(`<strong>${factor}</strong> (${interpretacion} al promedio)`);
            } else {
                explicaciones.push(`<strong>${factor}</strong>`);
            }
        });
        
        return explicaciones.join(', ') + '.';
    }

    // ===== ADVERTENCIA DE AMBIGÜEDAD =====
    function mostrarAdvertenciaAmbiguedad(resultado) {
        // Crear contenedor de advertencia si no existe
        let advertencia = $('.advertencia-ambiguedad');
        if (!advertencia) {
            advertencia = document.createElement('div');
            advertencia.className = 'advertencia-ambiguedad';
            $('#resultGama').parentNode.insertBefore(advertencia, $('#resultGama').nextSibling);
        }

        // Construir lista de gamas cercanas
        const gamasCercanas = resultado.clusters_cercanos
            .map(c => `<strong>${c.gama}</strong> (${(c.probabilidad * 100).toFixed(1)}%)`)
            .join(', ');

        advertencia.innerHTML = `
            <div class="alerta-warning">
                <i class="fas fa-exclamation-triangle"></i>
                <div>
                    <strong>⚠️ Dispositivo en zona límite</strong>
                    <p>Tu dispositivo está muy cerca de múltiples gamas: ${gamasCercanas}.</p>
                    <p>La clasificación puede variar según pequeños cambios en las especificaciones.</p>
                </div>
            </div>
        `;

        // Animar entrada
        anime({
            targets: advertencia,
            opacity: [0, 1],
            translateY: [-20, 0],
            duration: 600,
            easing: 'easeOutQuad'
        });
    }

    // ===== COMPARACIÓN USUARIO VS PROMEDIO =====
    function mostrarComparacionCompleta(datosUsuario, promedio) {
        const contenedorComparacion = document.createElement('div');
        contenedorComparacion.className = 'comparacion-usuario';
        contenedorComparacion.innerHTML = `
            <h3>Tu dispositivo vs Promedio de la gama</h3>
            <div class="tabla-comparacion">
                <div class="fila-comparacion encabezado">
                    <span>Característica</span>
                    <span>Tu valor</span>
                    <span>Promedio</span>
                </div>
                ${construirFilasComparacion(datosUsuario, promedio)}
            </div>
        `;
        
        // Insertar antes de los gráficos
        const primerGrafico = $('.chart-container');
        primerGrafico.parentNode.insertBefore(contenedorComparacion, primerGrafico);
    }

    function construirFilasComparacion(usuario, promedio) {
        const mapeoNombres = {
            'num_cores': 'Núcleos',
            'processor_speed': 'Velocidad (GHz)',
            'battery_capacity': 'Batería (mAh)',
            'fast_charging_available': 'Carga rápida',
            'ram_capacity': 'RAM (GB)',
            'internal_memory': 'Almacenamiento (GB)',
            'screen_size': 'Pantalla (pulg)',
            'resolution_width': 'Ancho de resolución',
            'resolution_height': 'Altura de resolución',
            'num_rear_cameras': 'Cámaras traseras',
            'primary_camera_rear': 'Cámara principal (MP)',
            'primary_camera_front': 'Cámara frontal (MP)',
            'price': 'Precio (USD)'
        };

        let html = '';
        for (const [clave, nombreLegible] of Object.entries(mapeoNombres)) {
            const valorUsuario = usuario[clave];
            const valorPromedio = promedio[nombreLegible] || '-';
            
            let valorUsuarioFormateado = valorUsuario;
            if (clave === 'fast_charging_available') {
                valorUsuarioFormateado = valorUsuario === '1' ? 'Sí' : 'No';
            } else if (clave === 'price') {
                valorUsuarioFormateado = `$${parseFloat(valorUsuario).toFixed(0)}`;
            }
            
            html += `
                <div class="fila-comparacion">
                    <span>${nombreLegible}</span>
                    <span class="valor-usuario">${valorUsuarioFormateado}</span>
                    <span class="valor-promedio">${valorPromedio}</span>
                </div>
            `;
        }
        
        return html;
    }

    // ===== GRÁFICO DE BARRAS (DIFERENCIAS RELATIVAS) =====
    function crearGraficoBarras(diferenciasRelativas) {
        if (graficoBarras) graficoBarras.destroy();
        
        const ctx = $('#chartDifRel').getContext('2d');
        const categorias = Object.keys(diferenciasRelativas);
        const valores = Object.values(diferenciasRelativas).map(v => v * 100);
        const etiquetas = categorias.map((cat, i) => `${cat} (${valores[i].toFixed(1)}%)`);

        graficoBarras = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: etiquetas,
                datasets: [{
                    label: 'Diferencia relativa (%)',
                    data: valores,
                    backgroundColor: valores.map(v =>
                        v >= 0 ? 'rgba(75, 192, 192, 0.7)' : 'rgba(255, 99, 132, 0.7)'
                    ),
                    borderColor: valores.map(v =>
                        v >= 0 ? 'rgba(75, 192, 192, 1)' : 'rgba(255, 99, 132, 1)'
                    ),
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: ctx => {
                                const valor = ctx.parsed.y;
                                const signo = valor >= 0 ? '+' : '';
                                return `${signo}${valor.toFixed(1)}% respecto al promedio`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { 
                            display: true, 
                            text: 'Diferencia (%)' 
                        }
                    }
                }
            }
        });
    }

    // ===== GRÁFICO DE DISPERSIÓN (PCA) =====
    function crearGraficoDispersion(resultado) {
        if (graficoDispersion) graficoDispersion.destroy();
        
        const ctx = $('#chartScatter').getContext('2d');
        
        // Colores por gama
        const coloresPorGama = {
            'Gama Alta': 'rgba(76, 175, 80, 0.8)',
            'Gama Media': 'rgba(255, 193, 7, 0.8)',
            'Gama Baja': 'rgba(244, 67, 54, 0.8)'
        };

        // Color del usuario - más brillante si está en zona ambigua
        const colorUsuario = resultado.es_ambiguo 
            ? 'rgba(255, 152, 0, 1)'  // Naranja para ambiguo
            : 'rgba(33, 150, 243, 1)'; // Azul para claro

        graficoDispersion = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [
                    // Regiones de fondo (sin interacción)
                    {
                        label: 'Áreas de gama',
                        data: resultado.pca_clusters.map((punto, indice) => ({ 
                            x: punto[0], 
                            y: punto[1] 
                        })),
                        backgroundColor: resultado.gamas.map(g => 
                            g === 'Gama Alta' ? 'rgba(76, 175, 80, 0.1)' :
                            g === 'Gama Media' ? 'rgba(255, 193, 7, 0.1)' :
                            'rgba(244, 67, 54, 0.1)'
                        ),
                        pointRadius: 80,
                        pointHoverRadius: 80,
                        borderWidth: 0,
                        hoverBackgroundColor: resultado.gamas.map(g => 
                            g === 'Gama Alta' ? 'rgba(76, 175, 80, 0.1)' :
                            g === 'Gama Media' ? 'rgba(255, 193, 7, 0.1)' :
                            'rgba(244, 67, 54, 0.1)'
                        ),
                        order: 0
                    },
                    // Centroides de clusters
                    {
                        label: 'Centroides de gamas',
                        data: resultado.pca_clusters.map((punto, indice) => ({
                            x: punto[0],
                            y: punto[1],
                            gama: resultado.gamas[indice],
                            probabilidad: resultado.probabilidades[resultado.gamas[indice]]
                        })),
                        backgroundColor: resultado.gamas.map(g => coloresPorGama[g]),
                        pointRadius: 12,
                        pointHoverRadius: 14,
                        borderColor: '#fff',
                        borderWidth: 2,
                        order: 1
                    },
                    // Punto del usuario
                    {
                        label: 'Tu dispositivo',
                        data: [{ 
                            x: resultado.pca_usuario[0], 
                            y: resultado.pca_usuario[1],
                            gama_predicha: resultado.gama,
                            es_ambiguo: resultado.es_ambiguo
                        }],
                        backgroundColor: colorUsuario,
                        pointRadius: 14,
                        pointHoverRadius: 16,
                        pointStyle: 'star',
                        borderColor: '#fff',
                        borderWidth: 3,
                        order: 999
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,  // FIXED: Permite mejor responsividad
                aspectRatio: 2,  // FIXED: Ratio de aspecto más apropiado
                interaction: {
                    mode: 'point',
                    intersect: true
                },
                plugins: {
                    datalabels: {
                        display: (context) => {
                            // Solo mostrar etiquetas para centroides
                            return context.datasetIndex === 1;
                        },
                        align: 'top',
                        offset: 8,
                        formatter: (valor) => valor.gama || '',
                        color: '#000',
                        font: { 
                            weight: 'bold', 
                            size: 12 
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: (contexto) => {
                                if (contexto.dataset.label === 'Tu dispositivo') {
                                    const lineas = [
                                        `Tu dispositivo: ${contexto.raw.gama_predicha}`,
                                        `Distancia al centroide: ${resultado.distancia_minima.toFixed(3)}`
                                    ];
                                    
                                    if (contexto.raw.es_ambiguo) {
                                        lineas.push('⚠️ En zona límite entre gamas');
                                    }
                                    
                                    // Agregar probabilidades
                                    lineas.push('');
                                    lineas.push('Probabilidades:');
                                    for (const [gama, prob] of Object.entries(resultado.probabilidades)) {
                                        lineas.push(`  ${gama}: ${(prob * 100).toFixed(1)}%`);
                                    }
                                    
                                    return lineas;
                                } else if (contexto.raw.gama) {
                                    const prob = (contexto.raw.probabilidad * 100).toFixed(1);
                                    return [
                                        `${contexto.raw.gama}`,
                                        `Probabilidad: ${prob}%`
                                    ];
                                }
                                return '';
                            }
                        },
                        filter: (tooltipItem) => {
                            // No mostrar tooltip para las regiones de fondo
                            return tooltipItem.datasetIndex !== 0;
                        }
                    },
                    legend: { 
                        position: 'top',
                        labels: {
                            filter: (legendItem) => {
                                // No mostrar en la leyenda las regiones de fondo
                                return legendItem.text !== 'Áreas de gama';
                            }
                        }
                    }
                },
                scales: {
                    x: { 
                        title: { 
                            display: true, 
                            text: 'Componente Principal 1' 
                        } 
                    },
                    y: { 
                        title: { 
                            display: true, 
                            text: 'Componente Principal 2' 
                        } 
                    }
                }
            },
            plugins: [ChartDataLabels]
        });
    }

    // ===== GRÁFICO DE DISTRIBUCIÓN (PIE) =====
    function crearGraficoDistribucion(distribucion) {
        if (graficoDistribucion) graficoDistribucion.destroy();
        
        const ctx = $('#chartDistribution').getContext('2d');
        const gamas = Object.keys(distribucion);
        const cantidades = Object.values(distribucion);
        const colores = ['#4CAF50', '#FFC107', '#F44336']; // Verde, Amarillo, Rojo
        const etiquetas = gamas.map((gama, i) => `${gama} (Cantidad: ${cantidades[i]})`);

        graficoDistribucion = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: etiquetas,
                datasets: [{
                    data: cantidades,
                    backgroundColor: colores,
                    borderColor: '#fff',
                    borderWidth: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { 
                        position: 'bottom',
                        labels: {
                            font: { size: 14 },
                            padding: 15
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: ctx => {
                                const total = cantidades.reduce((a, b) => a + b, 0);
                                const porcentaje = ((ctx.parsed / total) * 100).toFixed(1);
                                return `${ctx.label}: ${ctx.parsed} dispositivos (${porcentaje}%)`;
                            }
                        }
                    }
                }
            }
        });
    }

    // ===== REINICIAR APLICACIÓN =====
    $('#restartBtn').addEventListener('click', () => {
        // Recargar la página para reiniciar todo
        window.location.reload();
    });
});