// Previene que el WebView de Chromium intercepte los archivos arrastrados y los abra como
// recursos del navegador, dejando que el evento nativo de Tauri (FileDrop) sea capturado
// por el backend Rust antes de llegar al DOM.
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => e.preventDefault());

(function () {
  'use strict';

  const TIPS = [
    'Sabias que? YOLO significa "You Only Look Once" - la red analiza la imagen en una sola pasada!',
    'Consejo Python: Usa list comprehensions en vez de for-loops para codigo mas rapido y limpio.',
    'Dato curioso: Las redes neuronales se inspiran en como funcionan las neuronas del cerebro humano.',
    'Tip Gaming: En Counter-Strike, el "spray control" es como el data augmentation - practica hace al maestro!',
    'Sabias que? El "Transfer Learning" es como cuando usas lo aprendido en matemáticas para física!',
    'Consejo Python: Los f-strings (f"hola {nombre}") son mas rapidos que .format() o concatenacion.',
    'Dato curioso: YOLO26n-cls tiene solo 26 capas - perfecto para PCs de bajos recursos!',
    'Tip Dev: Ctrl+Z deshace en casi cualquier programa. Ctrl+Y lo rehace. Atajos salvadores!',
    'Sabias que? El "freeze" en entrenamiento congela capas antiguas para reutilizar conocimiento.',
    'Consejo Python: import this → El Zen de Python. Leelo, es corto y sabio.',
    'Dato curioso: Cada epoca es una vuelta COMPLETA al dataset de entrenamiento.',
    'Tip Gaming: En Minecraft, redstone es como programar: if/else con cables y antorchas!',
    'Sabias que? batch=4 significa que la red ve 4 imagenes juntas antes de corregirse.',
    'Consejo Python: Usa type hints (def fn(x: int) -> str) para codigo mas mantenible.',
    'Dato curioso: Data augmentation rota/escala imagenes para que la red aprenda mejor.',
    'Tip Dev: Comenta EL POR QUE haces algo, no LO QUE haces. El codigo ya dice lo que hace.',
    'Sabias que? imgsz=224 es el tamaño clasico usado en ImageNet, el dataset mas famoso del mundo.',
    'Consejo Python: Virtual environments evitan conflictos entre versiones de librerias. Usalos!',
    'Dato curioso: workers=2 usa 2 nucleos de CPU en paralelo para cargar imagenes mas rapido.',
    'Tip Gaming: En Fortnite, construir en 90 grados es habilidad motora fina, igual que dibujar datos!'
  ];

  const TOTAL_EPOCHS_FALLBACK = 15;

  const state = {
    classes: [],
    nextClassId: 1,
    currentTestImage: null,
    hasModel: false,
    bestPtPath: null,
    trainingTotalEpochs: TOTAL_EPOCHS_FALLBACK,
    tipTimerId: null,
    currentTipIndex: 0,
    isAdmin: false,
  };

  function el(id) {
    return document.getElementById(id);
  }

  // Debug reporter: no-op en producción. Para activar en desarrollo local,
  // reemplaza el cuerpo con la llamada a fetch al servidor de debug.
  // eslint-disable-next-line no-unused-vars
  function debugReport(_hypothesisId, _location, _msg, _data) {
    // no-op en producción
  }

  function getTauriGlobal() {
    return typeof window.__TAURI__ === 'object' ? window.__TAURI__ : null;
  }

  function getTauriInternals() {
    return typeof window.__TAURI_INTERNALS__ === 'object' ? window.__TAURI_INTERNALS__ : null;
  }

  function getTauriCore() {
    const tauri = getTauriGlobal();
    if (tauri && tauri.core && typeof tauri.core.invoke === 'function') {
      return tauri.core;
    }
    const internals = getTauriInternals();
    if (internals && typeof internals.invoke === 'function') {
      return {
        invoke: (cmd, args, options) => internals.invoke(cmd, args || {}, options),
      };
    }
    return null;
  }

  function invokeTauri(cmd, args) {
    const core = getTauriCore();
    if (core) {
      return core.invoke(cmd, args || {});
    }
    return Promise.reject(new Error('Tauri no disponible. Ejecuta en modo escritorio.'));
  }

  function hasTauri() {
    return !!getTauriCore();
  }

  async function listenTauriEvent(eventName, handler) {
    const tauri = getTauriGlobal();
    if (tauri && tauri.event && typeof tauri.event.listen === 'function') {
      return tauri.event.listen(eventName, handler);
    }

    const internals = getTauriInternals();
    if (!internals || typeof internals.transformCallback !== 'function') {
      throw new Error('API de eventos de Tauri no disponible');
    }

    const callbackId = internals.transformCallback(handler);
    const eventId = await invokeTauri('plugin:event|listen', {
      event: eventName,
      target: { kind: 'Any' },
      handler: callbackId,
    });

    return async function () {
      try {
        if (typeof internals.unregisterCallback === 'function') {
          internals.unregisterCallback(callbackId);
        }
      } catch (_) {}
      try {
        await invokeTauri('plugin:event|unlisten', {
          event: eventName,
          eventId: eventId,
        });
      } catch (_) {}
    };
  }

  function openDialog(options) {
    if (!hasTauri()) return Promise.resolve(null);
    const tauri = getTauriGlobal();
    try {
      if (tauri && tauri.dialog && tauri.dialog.open) {
        return tauri.dialog.open(options);
      }
    } catch (_) {}
    return invokeTauri('plugin:dialog|open', { options: options || {} });
  }

  function saveDialog(options) {
    if (!hasTauri()) return Promise.resolve(null);
    const tauri = getTauriGlobal();
    try {
      if (tauri && tauri.dialog && tauri.dialog.save) {
        return tauri.dialog.save(options);
      }
    } catch (_) {}
    return invokeTauri('plugin:dialog|save', { options: options || {} });
  }

  function showDialog(opts) {
    if (!hasTauri()) {
      alert(opts.message || opts.title || 'Mensaje');
      return Promise.resolve();
    }
    const tauri = getTauriGlobal();
    try {
      if (tauri && tauri.dialog && tauri.dialog.message) {
        return tauri.dialog.message(opts.message || '', {
          title: opts.title || 'DeepSight',
          kind: opts.kind || 'info',
        });
      }
    } catch (_) {}
    return invokeTauri('plugin:dialog|message', {
      message: opts.message || '',
      title: opts.title || 'DeepSight',
      kind: opts.kind || 'info',
    }).catch(() => {
      alert(opts.message || opts.title || '');
    });
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = String(str || '');
    return div.innerHTML;
  }

  function logLine(message, level, containerId) {
    const cid = containerId || 'terminal-output';
    const out = el(cid);
    if (!out) return;
    const div = document.createElement('div');
    div.className = 'log-line ' + (level || 'info');
    const time = new Date().toLocaleTimeString('es-ES', { hour12: false });
    const prefix = `[${time}] `;
    div.textContent = prefix + (message || '');
    out.appendChild(div);
    while (out.childElementCount > 500) {
      out.removeChild(out.firstChild);
    }
    out.scrollTop = out.scrollHeight;
  }

  function appendOverlayLog(message, level) {
    const out = el('overlay-terminal');
    if (!out) return;
    const div = document.createElement('div');
    div.className = 'log-line ' + (level || 'info');
    div.textContent = message || '';
    out.appendChild(div);
    while (out.childElementCount > 120) {
      out.removeChild(out.firstChild);
    }
    out.scrollTop = out.scrollHeight;
  }

  function clearOverlayLog() {
    const out = el('overlay-terminal');
    if (out) out.innerHTML = '';
  }

  function startTipCycler() {
    stopTipCycler();
    const tipText = el('overlay-tip-text');
    const tick = () => {
      const tip = TIPS[state.currentTipIndex % TIPS.length];
      state.currentTipIndex++;
      if (tipText) {
        tipText.style.opacity = '0';
        setTimeout(() => {
          tipText.textContent = tip;
          tipText.style.opacity = '1';
        }, 250);
      }
    };
    tick();
    state.tipTimerId = setInterval(tick, 4000);
  }

  function stopTipCycler() {
    if (state.tipTimerId) {
      clearInterval(state.tipTimerId);
      state.tipTimerId = null;
    }
  }

  function showTrainingOverlay() {
    const ov = el('training-overlay');
    if (!ov) return;
    ov.classList.remove('hidden');
    clearOverlayLog();
    updateOverlayEpoch(0, state.trainingTotalEpochs);
    startTipCycler();
  }

  function hideTrainingOverlay() {
    stopTipCycler();
    const ov = el('training-overlay');
    if (ov) ov.classList.add('hidden');
  }

  function updateOverlayEpoch(epoch, total) {
    const safeTotal = total || state.trainingTotalEpochs || TOTAL_EPOCHS_FALLBACK;
    const ep = el('overlay-epoch-text');
    const bar = el('overlay-epoch-bar');
    if (ep) ep.textContent = `${epoch} / ${safeTotal}`;
    const pct = safeTotal > 0 ? Math.min(100, Math.max(0, (epoch / safeTotal) * 100)) : 0;
    if (bar) bar.style.width = pct.toFixed(1) + '%';
  }

  function hideSplash() {
    const sp = el('splash-screen');
    if (!sp) return;
    sp.classList.add('fade-out');
  }

  function setHwBadge(info) {
    const txt = el('hw-info-text');
    const ind = document.querySelector('#hw-badge .status-indicator');
    if (!txt || !info) return;
    const cpu_cores = info.cpu_cores || '?';
    const ram_gb = info.total_memory_mb
      ? (info.total_memory_mb / 1024).toFixed(1) + 'GB'
      : '?';
    const low = info.is_low_spec ? ' · MODO CAFETERA' : '';
    txt.textContent = `${cpu_cores} nucleos · ${ram_gb} RAM · ${info.python_available ? 'Python ' + (info.python_version || '') : 'SIN PYTHON'}${low}`;
    if (ind) {
      ind.style.backgroundColor = info.python_available
        ? 'var(--accent-green)'
        : 'var(--accent-red)';
    }
  }

  function isAdminMode() {
    try {
      if (!hasTauri()) return false;
      return state.isAdmin;
    } catch (_) {
      return false;
    }
  }

  function setAdminBanner(show) {
    const b = el('admin-warning-banner');
    if (!b) return;
    if (show) b.classList.remove('hidden');
    else b.classList.add('hidden');
  }

  function createClassCard() {
    const id = 'cls_' + state.nextClassId;
    state.nextClassId++;
    const defName = 'Clase_' + state.classes.length;
    const cls = {
      id: id,
      name: defName,
      images: [],
    };
    state.classes.push(cls);

    const container = el('classes-container');
    const card = document.createElement('div');
    card.className = 'class-card';
    card.dataset.classId = id;

    const header = document.createElement('div');
    header.className = 'card-header';

    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'card-title-input';
    input.value = defName;
    input.maxLength = 40;
    input.placeholder = 'Nombre de clase...';
    input.addEventListener('input', () => {
      cls.name = input.value || ('Clase_' + state.classes.indexOf(cls));
      updateTrainBadge();
    });

    const del = document.createElement('button');
    del.className = 'btn-delete-class';
    del.title = 'Eliminar clase';
    del.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6l-2 14a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2L5 6"></path><path d="M10 11v6M14 11v6"></path><path d="M9 6V4a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2"></path></svg>';
    del.addEventListener('click', () => {
      if (!confirm(`Eliminar clase "${cls.name}"?`)) return;
      state.classes = state.classes.filter(c => c.id !== id);
      card.remove();
      updateTrainBadge();
    });

    header.appendChild(input);
    header.appendChild(del);

    const dz = document.createElement('div');
    dz.className = 'dropzone';
    dz.dataset.classDrop = id;
    dz.innerHTML = `
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
        <polyline points="17 8 12 3 7 8"></polyline>
        <line x1="12" y1="3" x2="12" y2="15"></line>
      </svg>
      <div class="dropzone-text">Arrastra imagenes aqui</div>
      <div class="dropzone-subtext">o haz clic para elegir</div>
    `;

    const fileList = document.createElement('div');
    fileList.className = 'file-list';
    fileList.dataset.fileList = id;

    const footer = document.createElement('div');
    footer.className = 'card-footer-info';
    const count = document.createElement('span');
    count.textContent = '0 imagenes';
    const badge = document.createElement('span');
    badge.className = 'count-badge';
    badge.textContent = 'min. 5';
    footer.appendChild(count);
    footer.appendChild(badge);

    card.appendChild(header);
    card.appendChild(dz);
    card.appendChild(fileList);
    card.appendChild(footer);
    container.appendChild(card);

    bindDropzone(dz, cls, fileList, count, badge);
    updateTrainBadge();
    container.scrollTop = container.scrollHeight;
    return cls;
  }

  function renderClassFile(cls, fileListEl, countEl, badgeEl) {
    fileListEl.innerHTML = '';
    // PARCHE M-01: Limitar renderizado DOM a 200 items para evitar congelamiento.
    // El estado interno (cls.images) conserva todos los paths.
    const MAX_VISIBLE = 200;
    const totalCount = cls.images.length;
    const visibleImages = cls.images.slice(0, MAX_VISIBLE);

    for (let i = 0; i < visibleImages.length; i++) {
      const path = visibleImages[i];
      const parts = path.replace(/\\/g, '/').split('/');
      const originalName = parts[parts.length - 1] || path;
      const item = document.createElement('div');
      item.className = 'file-item';
      item.innerHTML = `
        <div class="file-item-main">
          <svg class="file-item-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
            <circle cx="8.5" cy="8.5" r="1.5"></circle>
            <polyline points="21 15 16 10 5 21"></polyline>
          </svg>
          <span class="file-item-name"></span>
        </div>
        <button class="file-item-remove" title="Quitar">×</button>
      `;
      item.querySelector('.file-item-name').textContent = originalName;
      const capturedIndex = i;
      const rm = item.querySelector('.file-item-remove');
      rm.addEventListener('click', (ev) => {
        ev.stopPropagation();
        cls.images.splice(capturedIndex, 1);
        renderClassFile(cls, fileListEl, countEl, badgeEl);
        updateTrainBadge();
      });
      fileListEl.appendChild(item);
    }

    // Mostrar indicador de overflow si hay más de MAX_VISIBLE imágenes
    if (totalCount > MAX_VISIBLE) {
      const overflow = document.createElement('div');
      overflow.className = 'file-item';
      overflow.style.opacity = '0.6';
      overflow.style.fontStyle = 'italic';
      overflow.textContent = `... y ${totalCount - MAX_VISIBLE} imagen(es) más (no mostradas por rendimiento)`;
      fileListEl.appendChild(overflow);
    }

    updateClassFooter(cls, countEl, badgeEl);
  }

  function updateClassFooter(cls, countEl, badgeEl) {
    const n = cls.images.length;
    countEl.textContent = n + ' ' + (n === 1 ? 'imagen' : 'imagenes');
    if (badgeEl) {
      if (n >= 30) {
        badgeEl.textContent = 'MODO PRECISION';
        badgeEl.className = 'count-badge valid';
      } else if (n >= 5) {
        badgeEl.textContent = 'MODO RAPIDO';
        badgeEl.className = 'count-badge warning';
      } else {
        badgeEl.textContent = 'min. 5';
        badgeEl.className = 'count-badge';
      }
    }
  }

  function isImagePath(p) {
    if (!p) return false;
    const low = String(p).toLowerCase();
    return /\.(jpe?g|png|bmp|webp|tiff?)$/.test(low);
  }

  function getClassById(classId) {
    return state.classes.find((cls) => cls.id === classId) || null;
  }

  function getClassUiRefs(classId) {
    const card = document.querySelector(`.class-card[data-class-id="${classId}"]`);
    if (!card) return null;
    return {
      card,
      fileList: card.querySelector(`[data-file-list="${classId}"]`),
      countEl: card.querySelector('.card-footer-info span'),
      badgeEl: card.querySelector('.count-badge'),
      dropzone: card.querySelector('.dropzone'),
    };
  }

  async function addPathsToClassId(classId, paths) {
    const cls = getClassById(classId);
    const refs = getClassUiRefs(classId);
    if (!cls || !refs || !Array.isArray(paths) || paths.length === 0) return 0;

    // PARCHE M-01: Deduplicación O(1) con Set en lugar del indexOf O(n²)
    const existingSet = new Set(cls.images);
    const newPaths = [];

    // Procesar en chunks para no bloquear el hilo principal con 5000 archivos
    const CHUNK_SIZE = 200;
    for (let i = 0; i < paths.length; i += CHUNK_SIZE) {
      const chunk = paths.slice(i, i + CHUNK_SIZE);
      for (const path of chunk) {
        if (typeof path !== 'string' || !isImagePath(path)) continue;
        if (existingSet.has(path)) continue;
        existingSet.add(path);
        newPaths.push(path);
      }
      // Yield al event loop cada chunk para evitar congelamiento
      if (i + CHUNK_SIZE < paths.length) {
        await new Promise((r) => setTimeout(r, 0));
      }
    }

    if (newPaths.length > 0) {
      cls.images.push(...newPaths);
      renderClassFile(cls, refs.fileList, refs.countEl, refs.badgeEl);
      updateTrainBadge();
      logLine(`[Clase "${cls.name}"] ${newPaths.length} imagen(es) agregadas. Total: ${cls.images.length}`, 'info');
    }

    return newPaths.length;
  }


  function findDropzoneClassIdFromPosition(position) {
    if (!position) return null;
    const elements = document.elementsFromPoint(position.x, position.y);
    for (const element of elements) {
      const dropzone = element.closest ? element.closest('.dropzone[data-class-drop]') : null;
      if (dropzone && dropzone.dataset.classDrop) {
        return dropzone.dataset.classDrop;
      }
    }
    return null;
  }

  function collectPathsFromValue(value, paths) {
    if (!value) return;
    if (typeof value === 'string') {
      paths.push(value);
      return;
    }
    if (Array.isArray(value)) {
      value.forEach((item) => collectPathsFromValue(item, paths));
      return;
    }
    if (typeof value === 'object') {
      if (typeof value.path === 'string') {
        paths.push(value.path);
      }
      if (Array.isArray(value.paths)) {
        value.paths.forEach((item) => collectPathsFromValue(item, paths));
      }
      if (Array.isArray(value.files)) {
        value.files.forEach((item) => collectPathsFromValue(item, paths));
      }
      if (value.payload) {
        collectPathsFromValue(value.payload, paths);
      }
      if (value.detail) {
        collectPathsFromValue(value.detail, paths);
      }
    }
  }

  function extractDroppedPaths(e) {
    const rawPaths = [];

    try {
      const dt = e && e.dataTransfer;
      if (dt && dt.files) {
        Array.from(dt.files).forEach((file) => {
          if (file && typeof file.path === 'string') {
            rawPaths.push(file.path);
          }
        });
      }

      if (dt && dt.items) {
        Array.from(dt.items).forEach((item) => {
          const file = item && typeof item.getAsFile === 'function' ? item.getAsFile() : null;
          if (file && typeof file.path === 'string') {
            rawPaths.push(file.path);
          }
        });
      }
    } catch (_) {}

    collectPathsFromValue(e && e.detail, rawPaths);
    collectPathsFromValue(e && e.payload, rawPaths);

    return Array.from(new Set(rawPaths.filter((p) => typeof p === 'string' && p.trim().length > 0)));
  }

  function bindDropzone(dz, cls, fileList, countEl, badgeEl) {
    dz.addEventListener('click', async () => {
      if (isAdminMode()) {
        showDialog({
          title: 'Modo Administrador',
          message: 'El selector de archivos esta deshabilitado en modo Administrador. Ejecuta la app sin privilegios para elegir archivos.',
          kind: 'warning',
        });
        return;
      }
      try {
        const sel = await openDialog({
          multiple: true,
          filters: [{
            name: 'Imagenes',
            extensions: ['jpg', 'jpeg', 'png', 'bmp', 'webp', 'tif', 'tiff'],
          }],
        });
        if (Array.isArray(sel)) await addPathsToClassId(cls.id, sel);
        else if (typeof sel === 'string') await addPathsToClassId(cls.id, [sel]);
      } catch (e) {
        logLine('Error abriendo dialogo: ' + e.message, 'error');
      }
    });

    dz.addEventListener('dragenter', (e) => {
      e.preventDefault();
      e.stopPropagation();
      dz.classList.add('dragover');
    });
    dz.addEventListener('dragover', (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy';
      dz.classList.add('dragover');
    });
    dz.addEventListener('dragleave', (e) => {
      e.preventDefault();
      e.stopPropagation();
      dz.classList.remove('dragover');
    });
    dz.addEventListener('drop', async (e) => {
      e.preventDefault();
      e.stopPropagation();
      dz.classList.remove('dragover');
      if (isAdminMode()) {
        showDialog({
          title: 'Modo Administrador',
          message: 'Arrastrar y soltar esta inhabilitado en Modo Administrador por seguridad de Windows (UIPI). Cierra y abre la app normalmente para usar Drag & Drop.',
          kind: 'warning',
        });
        return;
      }
      let paths = [];
      try {
        paths = extractDroppedPaths(e);
      } catch (_) {}
      await addPathsToClassId(cls.id, paths);
    });
  }

  async function setupNativeDragDrop() {
    if (!hasTauri()) return;

    const body = document.body;
    if (body) {
      body.addEventListener('dragover', (e) => {
        e.preventDefault();
      });
    }

    try {
      const clearDragState = () => {
        document.querySelectorAll('.dropzone.dragover').forEach((node) => {
          node.classList.remove('dragover');
        });
      };

      const setDragState = (position) => {
        const classId = findDropzoneClassIdFromPosition(position);
        clearDragState();
        if (!classId) return null;
        const refs = getClassUiRefs(classId);
        if (refs && refs.dropzone) {
          refs.dropzone.classList.add('dragover');
        }
        return classId;
      };

      await listenTauriEvent('tauri://drag-enter', (event) => {
        const payload = event && event.payload;
        if (!payload) return;
        setDragState(payload.position);
      });

      await listenTauriEvent('tauri://drag-over', (event) => {
        const payload = event && event.payload;
        if (!payload) return;
        setDragState(payload.position);
      });

      await listenTauriEvent('tauri://drag-leave', () => {
        clearDragState();
      });

      await listenTauriEvent('tauri://drag-drop', async (event) => {
        const payload = event && event.payload;
        clearDragState();
        if (!payload) return;
        const classId = findDropzoneClassIdFromPosition(payload.position);
        if (!classId) return;
        await addPathsToClassId(classId, payload.paths || []);
      });

      // Canal secundario robusto: rutas absolutas emitidas desde el handler
      // WindowEvent::DragDrop de Rust. Se usa como fallback cuando tauri://drag-drop
      // no entrega las rutas (race condition entre WebView y el OS en Windows).
      let lastDropPosition = null;
      await listenTauriEvent('tauri://drag-over', (event) => {
        const pos = event && event.payload && event.payload.position;
        if (pos) lastDropPosition = pos;
      });
      await listenTauriEvent('file-drop-paths', async (event) => {
        const paths = event && event.payload;
        clearDragState();
        if (!Array.isArray(paths) || paths.length === 0) return;

        // --- Enrutador por coordenadas ---
        // Comprueba si la posición del drop cae dentro del panel de inferencia.
        // getBoundingClientRect() devuelve coordenadas en espacio de viewport CSS,
        // que coinciden con las coordenadas lógicas que Tauri reporta en Windows.
        const testDz = el('test-dropzone');
        if (testDz && lastDropPosition) {
          const rect = testDz.getBoundingClientRect();
          const { x, y } = lastDropPosition;
          if (x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom) {
            // Drop sobre el panel de inferencia: cargar imagen y ejecutar inferencia
            const imagePath = paths.find(isImagePath);
            if (imagePath && loadTestImagePath(imagePath) && state.hasModel) {
              await runInference();
            } else if (imagePath) {
              loadTestImagePath(imagePath);
              logLine('Imagen de prueba cargada. Entrena un modelo para ejecutar inferencia.', 'info');
            }
            return; // No continuar hacia la lógica de clases
          }
        }

        // Drop fuera del panel de inferencia: enrutar a clases de entrenamiento
        const classId = findDropzoneClassIdFromPosition(lastDropPosition) || null;
        if (classId) {
          await addPathsToClassId(classId, paths);
        } else {
          // No hay dropzone de clase bajo el cursor: agregar a la primera clase disponible
          const firstClass = state.classes[0];
          if (firstClass) await addPathsToClassId(firstClass.id, paths);
        }
      });
    } catch (e) {
      logLine('Listener nativo DragDrop no disponible: ' + e.message, 'warning');
    }
  }

  function updateTrainBadge() {
    const dot = document.querySelector('#train-mode-badge .badge-dot');
    const txt = el('train-mode-text');
    const btn = el('btn-train');
    if (!dot || !txt || !btn) return;

    const total = state.classes.reduce((s, c) => s + c.images.length, 0);
    const validClasses = state.classes.filter(c => c.images.length >= 1 && c.name.trim().length > 0);

    if (state.classes.length === 0) {
      dot.style.backgroundColor = 'var(--text-muted)';
      txt.textContent = 'Crea al menos 1 clase';
      btn.disabled = true;
      return;
    }
    if (state.classes.length < 2) {
      dot.style.backgroundColor = 'var(--accent-amber)';
      txt.textContent = 'Minimo 2 clases para entrenar';
      btn.disabled = true;
      return;
    }
    if (validClasses.length < 2) {
      dot.style.backgroundColor = 'var(--accent-amber)';
      txt.textContent = 'Agrega imagenes a tus clases';
      btn.disabled = true;
      return;
    }

    const minPerClass = Math.min.apply(null, validClasses.map(c => c.images.length));

    if (minPerClass >= 30) {
      dot.style.backgroundColor = 'var(--accent-green)';
      txt.textContent = `Modo Precision (≥30 img/clase) · ${total} totales`;
    } else if (minPerClass >= 5) {
      dot.style.backgroundColor = 'var(--accent-blue)';
      txt.textContent = `Modo Rapido (<30 img/clase) · ${total} totales · augmentation agresivo`;
    } else {
      dot.style.backgroundColor = 'var(--accent-amber)';
      txt.textContent = `Muy pocas imagenes (recomendado ≥5/clase) · ${total} totales`;
    }
    btn.disabled = false;
  }

  // Carga una ruta de imagen en el panel de inferencia (test) actualizando el estado
  // y la UI del badge. Retorna true si la ruta es válida y se cargó correctamente.
  // Expuesta a nivel de módulo para que el listener nativo file-drop-paths pueda
  // enrutar drops sobre #test-dropzone sin acceder al scope de bindTestDropzone.
  function loadTestImagePath(path) {
    if (!path || !isImagePath(path)) return false;
    state.currentTestImage = path;
    const parts = path.replace(/\\/g, '/').split('/');
    const name = parts[parts.length - 1] || path;
    const badge = el('test-file-badge');
    const fname = el('test-file-name');
    if (badge) badge.classList.remove('hidden');
    if (fname) fname.textContent = name;

    // --- Miniatura con protocolo seguro de assets de Tauri ---
    // convertFileSrc() traduce una ruta absoluta del sistema de archivos (ej. C:\...\foto.jpg)
    // al protocolo interno 'asset://' que el WebView puede cargar sin violar CSP.
    // Requiere: withGlobalTauri:true en tauri.conf.json y assetProtocol.enable:true.
    const thumbWrap = el('test-thumbnail-wrap');
    const thumb = el('test-thumbnail');
    if (thumbWrap && thumb) {
      try {
        const tauri = window.__TAURI__;
        const convertFn =
          (tauri && tauri.core && tauri.core.convertFileSrc) ||
          (tauri && tauri.convertFileSrc) ||
          null;
        if (convertFn) {
          const safeUrl = convertFn(path);
          thumb.classList.remove('loaded');
          thumb.src = safeUrl;
          thumb.onload = () => thumb.classList.add('loaded');
          thumb.onerror = () => {
            // Fallback silencioso: ocultar el wrap si la URL no se puede cargar
            thumbWrap.classList.add('hidden');
          };
          thumbWrap.classList.remove('hidden');
        }
      } catch (_) {
        // Si convertFileSrc no está disponible (modo navegador), ignorar la miniatura
      }
    }

    return true;
  }

  function bindTestDropzone() {
    const dz = el('test-dropzone');
    if (!dz) return;

    // Alias local — delega en la función de módulo para coherencia
    const setTestImage = loadTestImagePath;

    dz.addEventListener('click', async () => {
      if (isAdminMode()) {
        showDialog({
          title: 'Modo Administrador',
          message: 'Usa el modo normal para cargar imagenes de prueba.',
          kind: 'warning',
        });
        return;
      }
      try {
        const sel = await openDialog({
          multiple: false,
          filters: [{
            name: 'Imagen',
            extensions: ['jpg', 'jpeg', 'png', 'bmp', 'webp', 'tif', 'tiff'],
          }],
        });
        if (typeof sel === 'string' && setTestImage(sel)) {
          if (state.hasModel) await runInference();
        }
      } catch (e) {
        logLine('Error: ' + e.message, 'error');
      }
    });

    ['dragenter', 'dragover'].forEach(ev => {
      dz.addEventListener(ev, (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (ev === 'dragover' && e.dataTransfer) e.dataTransfer.dropEffect = 'copy';
        dz.classList.add('dragover');
      });
    });
    ['dragleave', 'drop'].forEach(ev => {
      dz.addEventListener(ev, (e) => {
        e.preventDefault(); e.stopPropagation(); dz.classList.remove('dragover');
      });
    });
    dz.addEventListener('drop', async (e) => {
      if (isAdminMode()) {
        showDialog({
          title: 'Modo Administrador',
          message: 'Drag & Drop inhabilitado en modo Administrador.',
          kind: 'warning',
        });
        return;
      }
      try {
        const paths = extractDroppedPaths(e);
        if (paths[0] && setTestImage(paths[0]) && state.hasModel) {
          await runInference();
        }
      } catch (_) {}
    });
  }

  async function runInference() {
    if (!state.currentTestImage) return;
    if (!state.hasModel) {
      showDialog({
        title: 'Sin modelo',
        message: 'Entrena primero un modelo para realizar inferencias.',
        kind: 'warning',
      });
      return;
    }
    logLine('Iniciando inferencia...', 'info');
    const resName = el('res-class-name');
    const resConf = el('res-confidence-text');
    const bar = el('confidence-bar-fill');
    if (resName) resName.textContent = 'Analizando...';
    if (resConf) resConf.textContent = '---';
    if (bar) bar.style.width = '0%';

    try {
      const r = await invokeTauri('run_test_inference', {
        imagePath: state.currentTestImage,
      });
      if (r && r.success && r.class_name) {
        if (resName) resName.textContent = r.class_name;
        const conf = typeof r.confidence === 'number' ? r.confidence : 0;
        if (resConf) resConf.textContent = conf.toFixed(1) + '%';
        if (bar) bar.style.width = Math.min(100, conf).toFixed(1) + '%';
        logLine(`Inferencia OK → ${r.class_name} (${conf.toFixed(1)}%)`, 'success');
      } else {
        const err = (r && r.error_message) || 'Inferencia fallida';
        if (resName) resName.textContent = 'Error';
        if (resConf) resConf.textContent = '---';
        logLine('Inferencia fallida: ' + err, 'error');
        showDialog({
          title: 'Error de inferencia',
          message: err,
          kind: 'error',
        });
      }
    } catch (e) {
      if (resName) resName.textContent = 'Error';
      logLine('Invoke fallo: ' + e.message, 'error');
      showDialog({
        title: 'Error',
        message: e.message || String(e),
        kind: 'error',
      });
    }
  }

  function bindTrainButton() {
    const btn = el('btn-train');
    if (!btn) return;
    // PARCHE A-02: flag de guardia local para cerrar la ventana de race condition
    // entre el click y el momento en que Rust activa is_training en su Mutex.
    let _isTrainingInFlight = false;

    btn.addEventListener('click', async () => {
      // Bloqueo inmediato antes del invoke async para prevenir doble-disparo
      if (_isTrainingInFlight) return;

      if (!hasTauri()) {
        alert('Tauri no disponible. Para entrenar ejecuta: cargo tauri dev');
        return;
      }
      const classesToSend = state.classes
        .filter(c => c.name.trim().length > 0 && c.images.length > 0)
        .map(c => ({ name: c.name.trim(), images: c.images.slice() }));

      if (classesToSend.length < 2) {
        showDialog({
          title: 'Faltan datos',
          message: 'Necesitas al menos 2 clases con al menos 1 imagen cada una.',
          kind: 'warning',
        });
        return;
      }

      // Marcar in-flight Y deshabilitar visualmente ANTES del invoke
      _isTrainingInFlight = true;
      btn.disabled = true;

      state.trainingTotalEpochs = TOTAL_EPOCHS_FALLBACK;
      updateOverlayEpoch(0, state.trainingTotalEpochs);
      showTrainingOverlay();
      appendOverlayLog('Enviando dataset al motor Rust + Python...', 'info');

      try {
        const res = await invokeTauri('start_training', {
          classes: classesToSend,
        });
        if (!res || !res.success) {
          // El invoke respondió con error: restaurar botón
          _isTrainingInFlight = false;
          hideTrainingOverlay();
          showDialog({
            title: 'No se pudo iniciar',
            message: (res && res.message) || 'Error desconocido',
            kind: 'error',
          });
          // Restaurar estado del botón según validez de datos
          updateTrainBadge();
        } else {
          appendOverlayLog('Proceso iniciado correctamente.', 'success');
          logLine('Entrenamiento iniciado en segundo plano.', 'info');
          // _isTrainingInFlight se resetea cuando llega training:complete
        }
      } catch (e) {
        _isTrainingInFlight = false;
        hideTrainingOverlay();
        appendOverlayLog('Error: ' + e.message, 'error');
        logLine('Error iniciando entrenamiento: ' + e.message, 'error');
        showDialog({
          title: 'Error iniciando entrenamiento',
          message: e.message || String(e),
          kind: 'error',
        });
        updateTrainBadge();
      }
    });

    // Resetear el flag in-flight cuando el entrenamiento termina (éxito o error)
    // Nos suscribimos al evento global via una función accesible al scope
    window._resetTrainingInFlight = () => {
      _isTrainingInFlight = false;
    };
  }


  function bindExportButton() {
    const btn = el('btn-export');
    if (!btn) return;
    btn.addEventListener('click', async () => {
      if (!state.bestPtPath) {
        showDialog({
          title: 'Sin modelo',
          message: 'Aun no hay un modelo entrenado para exportar.',
          kind: 'warning',
        });
        return;
      }
      try {
        const savePath = await saveDialog({
          title: 'Guardar modelo entrenado',
          defaultPath: 'best.pt',
          filters: [{ name: 'Modelo PyTorch', extensions: ['pt'] }],
        });
        if (!savePath) return;
        const tauri = getTauriGlobal();
        const fs = tauri && tauri.fs;
        let success = false;
        if (fs && fs.copyFile) {
          try {
            await fs.copyFile(state.bestPtPath, savePath);
            success = true;
          } catch (_) {}
        }
        if (!success && tauri && tauri.shell && tauri.shell.Command) {
          try {
            const cmd = new tauri.shell.Command(
              'cmd',
              ['/c', 'copy', '/Y', state.bestPtPath, savePath]
            );
            await cmd.execute();
            success = true;
          } catch (_) {}
        }
        if (!success) {
          success = await invokeTauri('copy_best_pt', { from: state.bestPtPath, to: savePath })
            .then(() => true)
            .catch(() => false);
        }
        if (success) {
          logLine('Modelo exportado a: ' + savePath, 'success');
          showDialog({
            title: 'Exportado!',
            message: 'El modelo best.pt se guardo correctamente.\n\nRuta: ' + savePath,
            kind: 'info',
          });
        } else {
          logLine('No se pudo copiar automaticamente. Ruta del best.pt: ' + state.bestPtPath, 'warning');
          showDialog({
            title: 'Acceso manual',
            message:
              'El exportador no pudo copiar el archivo automaticamente.\n\n' +
              'Copia manualmente el archivo desde:\n' +
              state.bestPtPath +
              '\n\na la carpeta que prefieras.',
            kind: 'warning',
          });
        }
      } catch (e) {
        showDialog({
          title: 'Error exportando',
          message: e.message || String(e),
          kind: 'error',
        });
      }
    });
  }

  function bindClearLog() {
    const b = el('btn-clear-log');
    if (!b) return;
    b.addEventListener('click', () => {
      const out = el('terminal-output');
      if (out) out.innerHTML = '<div class="log-line info">[SYS] Consola limpiada.</div>';
    });
  }

  function bindThemeToggle() {
    const sw = el('theme-toggle-checkbox');
    if (sw) {
      sw.addEventListener('change', () => {
        if (sw.checked) {
          document.body.classList.remove('theme-dark');
          document.body.classList.add('theme-light');
        } else {
          document.body.classList.remove('theme-light');
          document.body.classList.add('theme-dark');
        }
      });
    }
  }

  function setupTauriListeners() {
    if (!hasTauri()) return;

    const attach = (eventName, handler) => {
      listenTauriEvent(eventName, handler).catch((e) => {
        logLine(`Listener ${eventName} fallo: ${e.message}`, 'warning');
      });
    };

    attach('training:log', (evt) => {
      const p = evt && evt.payload;
      if (!p) return;
      const content = p.content || {};
      const level = (content.level) || 'info';
      const msg = (content.message) || p.raw || '';
      logLine(msg, level);
      appendOverlayLog(msg, level);
    });

    // PARCHE A-03: Listener para el batch de logs (rate-limited).
    // Rust ahora agrupa los eventos 'log' y 'raw' en batches de 150ms.
    attach('training:log_batch', (evt) => {
      const batch = evt && evt.payload;
      if (!Array.isArray(batch)) return;
      for (const p of batch) {
        if (!p) continue;
        const content = p.content || {};
        const level = content.level || 'info';
        const msg = content.message || p.raw || '';
        if (msg.trim()) {
          logLine(msg, level);
          appendOverlayLog(msg, level);
        }
      }
    });


    attach('training:progress', (evt) => {
      const p = evt && evt.payload;
      if (!p) return;
      const c = p.content || {};
      const epoch = typeof c.epoch === 'number' ? c.epoch : 0;
      const total = typeof c.total_epochs === 'number' ? c.total_epochs : state.trainingTotalEpochs;
      if (total && total !== state.trainingTotalEpochs) {
        state.trainingTotalEpochs = total;
      }
      updateOverlayEpoch(epoch, state.trainingTotalEpochs);
      logLine(`Progreso epoca ${epoch}/${state.trainingTotalEpochs}`, 'info');
    });

    attach('training:hyperparameters', (evt) => {
      const p = evt && evt.payload;
      if (!p || !p.content) return;
      const data = p.content.data || {};
      if (typeof data.epochs === 'number') {
        state.trainingTotalEpochs = data.epochs;
        updateOverlayEpoch(0, data.epochs);
      }
      const mode = data.mode_name || data.augmentation_mode || '?';
      logLine(`Hiperparametros: epochs=${data.epochs || '?'}, freeze=${data.freeze || '?'}, modo=${mode}`, 'success');
      appendOverlayLog(`Hiperparametros: epochs=${data.epochs || '?'}, freeze=${data.freeze || '?'}, modo=${mode}`, 'success');
    });

    attach('training:error', (evt) => {
      const p = evt && evt.payload;
      if (!p) return;
      const msg = (p.content && p.content.message) || p.raw || 'Error entrenamiento';
      logLine(msg, 'error');
      appendOverlayLog(msg, 'error');
    });

    attach('training:raw', (evt) => {
      const p = evt && evt.payload;
      if (!p) return;
      const raw = p.raw || '';
      if (raw.trim()) {
        appendOverlayLog(raw, 'info');
        logLine(raw, 'info');
      }
    });

    attach('training:complete', (evt) => {
      const r = evt && evt.payload;
      if (!r) return;
      // PARCHE A-02: Resetear el guard de in-flight para permitir nuevos entrenamientos
      if (typeof window._resetTrainingInFlight === 'function') {
        window._resetTrainingInFlight();
      }
      setTimeout(() => {
        hideTrainingOverlay();
        if (r.success && r.best_pt_path) {
          state.hasModel = true;
          state.bestPtPath = r.best_pt_path;
          const exp = el('btn-export');
          if (exp) exp.disabled = false;
          logLine('ENTRENAMIENTO FINALIZADO! Modelo en: ' + r.best_pt_path, 'success');
          showDialog({
            title: 'Modelo listo!',
            message:
              'Entrenamiento completado exitosamente.\n\n' +
              'Ahora puedes arrastrar una imagen en la seccion "Prueba" para probarlo.\n\n' +
              'Ruta del modelo:\n' + r.best_pt_path,
            kind: 'info',
          });
          if (state.currentTestImage) runInference();
        } else {
          state.hasModel = false;
          const err = r.error_message || 'Ocurrio un error durante el entrenamiento.';
          logLine('ENTRENAMIENTO FALLIDO: ' + err, 'error');
          showDialog({
            title: 'Entrenamiento fallido',
            message: err,
            kind: 'error',
          });
        }
        // Restaurar estado visual del botón de entrenamiento
        updateTrainBadge();
      }, 600);
    });


    attach('training:json_complete', (evt) => {
      const p = evt && evt.payload;
      if (!p || !p.content) return;
      const c = p.content;
      if (c.best_pt_path && !state.bestPtPath) {
        state.bestPtPath = c.best_pt_path;
      }
    });
  }

  async function boot() {
    const hwTxt = el('hw-info-text');
    if (hwTxt) hwTxt.textContent = 'Iniciando...';
    bindThemeToggle();
    bindClearLog();
    bindTrainButton();
    bindExportButton();
    bindTestDropzone();
    setupTauriListeners();
    setupNativeDragDrop();

    document.documentElement.dataset.tauriBridge = hasTauri() ? 'ready' : 'missing';

    const splashDelay = hasTauri() ? 1200 : 500;
    setTimeout(() => {
      hideSplash();
    }, splashDelay);

    if (hasTauri()) {
      try {
        const info = await invokeTauri('get_system_info');
        setHwBadge(info);
        if (!info.python_available) {
          setTimeout(() => {
            showDialog({
              title: 'Python no detectado',
              message:
                'No se detecto Python 3.10+ en el PATH del sistema.\n\n' +
                'Para entrenar modelos necesitas instalar Python desde:\n' +
                'https://www.python.org/downloads/\n\n' +
                'IMPORTANTE: Marca "Add Python to PATH" durante la instalacion.\n\n' +
                'Luego instala dependencias:\n' +
                'pip install ultralytics',
              kind: 'error',
            });
          }, 700);
        }
      } catch (e) {
        logLine('Error obteniendo info del sistema: ' + e.message, 'warning');
        if (hwTxt) hwTxt.textContent = 'No se pudo detectar hardware';
      }

      try {
        setAdminBanner(state.isAdmin);
      } catch (_) {}
    } else {
      if (hwTxt) hwTxt.textContent = 'Modo navegador (sin Tauri)';
    }

    el('btn-add-class').addEventListener('click', () => createClassCard());
    createClassCard();
    createClassCard();

    logLine('DeepSight v2.0 - Listo para operar.', 'success');
    logLine('Paso 1: Define al menos 2 clases. Paso 2: Arrastra imagenes. Paso 3: Entrena!', 'info');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
