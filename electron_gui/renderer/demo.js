const state = { config: null, inputPath: '', running: false, streamedResults: 0, projects: [] };
const PROJECT_OPTIONS = [
  ['baseline', '纯 Qwen Baseline'],
  ['so', '空间特征加速'],
  ['codec', '帧间冗余加速'],
];

const $ = (id) => document.getElementById(id);
const ms = (value) => `${Number(value || 0).toFixed(1)} ms`;
const kb = (value) => `${(Number(value || 0) / 1024).toFixed(2)} KB`;

function serverUrl(config) {
  let host = config.serverAddress || (config.serverHost === '0.0.0.0' ? '127.0.0.1' : config.serverHost);
  if (host === 'localhost') host = '127.0.0.1';
  return `http://${host}:${config.serverPort}`;
}

function setServerStatus(value) {
  const warming = typeof value === 'object' && value.warming;
  const online = typeof value === 'object' ? value.running : value;
  const element = $('server-state');
  element.className = `state ${online ? 'online' : 'offline'}`;
  element.textContent = warming ? '云端预热中' : (online ? '云端在线' : '云端未启动');
  $('server-button').textContent = online ? '停止云端' : '启动云端';
  if (warming) $('run-status').textContent = '首次测试正在预热云端模型，预热时间不计入结果…';
}

function renderProjects() {
  const container = $('projects-list');
  container.innerHTML = '';
  if (!state.projects.length) {
    container.innerHTML = '<div class="empty project-empty">尚未添加项目。</div>';
    return;
  }
  state.projects.forEach((project, index) => {
    const row = document.createElement('div'); row.className = 'project-row';
    const number = document.createElement('span'); number.className = 'project-number'; number.textContent = `${index + 1}`;
    const select = document.createElement('select'); select.className = 'project-select';
    PROJECT_OPTIONS.forEach(([value, label]) => { const option = document.createElement('option'); option.value = value; option.textContent = label; select.append(option); });
    select.value = project;
    select.addEventListener('change', () => { state.projects[index] = select.value; });
    const up = document.createElement('button'); up.className = 'button project-button'; up.textContent = '↑'; up.disabled = index === 0;
    up.addEventListener('click', () => { [state.projects[index - 1], state.projects[index]] = [state.projects[index], state.projects[index - 1]]; renderProjects(); });
    const down = document.createElement('button'); down.className = 'button project-button'; down.textContent = '↓'; down.disabled = index === state.projects.length - 1;
    down.addEventListener('click', () => { [state.projects[index], state.projects[index + 1]] = [state.projects[index + 1], state.projects[index]]; renderProjects(); });
    const remove = document.createElement('button'); remove.className = 'button project-button'; remove.textContent = '×';
    remove.addEventListener('click', () => { state.projects.splice(index, 1); renderProjects(); });
    row.append(number, select, up, down, remove); container.append(row);
  });
}

function loadSettingsForm(config) {
  const values = {
    python: config.pythonPath || '', cloudCheckpoint: config.cloudCheckpoint || '', edgeCheckpoint: config.edgeCheckpoint || '',
    qwen: config.qwenPath || '', serverAddress: config.serverAddress || 'localhost', host: config.serverHost || '0.0.0.0', port: config.serverPort || 8080,
    device: config.device || 'cuda', timeout: config.timeout || 300, maxFrames: config.maxFrames || 8, sampleFps: config.sampleFps || 2,
    spatialLevel: config.spatialLevel || '49x64', rawWidth: config.rawWidth || 224, rawHeight: config.rawHeight || 224,
    rawFps: config.rawFps || 10, rawFormat: config.rawFormat || 'rgb24', offline: !!config.offlineMode, preload: config.preloadQwen !== false,
  };
  Object.entries(values).forEach(([key, value]) => {
    const element = $(`setting-${key.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`)}`);
    if (!element) return;
    if (element.type === 'checkbox') element.checked = value;
    else element.value = value;
  });
}

function readSettingsForm() {
  return {
    pythonPath: $('setting-python').value.trim(), cloudCheckpoint: $('setting-cloud-checkpoint').value.trim(), edgeCheckpoint: $('setting-edge-checkpoint').value.trim(),
    qwenPath: $('setting-qwen').value.trim(), serverAddress: $('setting-server-address').value.trim(), serverHost: $('setting-host').value.trim(), serverPort: Number($('setting-port').value),
    device: $('setting-device').value, timeout: Number($('setting-timeout').value), maxFrames: Number($('setting-max-frames').value), sampleFps: Number($('setting-sample-fps').value),
    spatialLevel: $('setting-spatial-level').value.trim() || '49x64', rawWidth: Number($('setting-raw-width').value), rawHeight: Number($('setting-raw-height').value),
    rawFps: Number($('setting-raw-fps').value), rawFormat: $('setting-raw-format').value, offlineMode: $('setting-offline').checked, preloadQwen: $('setting-preload').checked,
  };
}

async function saveSettings() {
  state.config = { ...state.config, ...readSettingsForm() };
  const result = await window.electronAPI.saveConfig(state.config);
  $('settings-message').textContent = result.success ? '已保存' : '保存失败';
  if (result.success) $('settings-modal').classList.add('hidden');
}

async function startOrStopServer() {
  if ((await window.electronAPI.checkCloudServerStatus()).running) {
    await window.electronAPI.stopCloudServer();
    return;
  }
  if (!state.config.cloudCheckpoint) {
    $('settings-modal').classList.remove('hidden');
    $('settings-message').textContent = '请先设置云端 checkpoint';
    return;
  }
  $('run-status').textContent = '正在启动并预热云端…';
  const result = await window.electronAPI.startCloudServer({
    checkpoint: state.config.cloudCheckpoint, host: state.config.serverHost, port: state.config.serverPort,
    device: state.config.device, qwenPath: state.config.qwenPath, offlineMode: state.config.offlineMode, preloadQwen: state.config.preloadQwen !== false,
  });
  $('run-status').textContent = result.success ? '云端已就绪' : `启动失败：${result.error}`;
}

async function chooseInput() {
  const result = await window.electronAPI.selectFile({ filters: [{ name: '媒体与裸帧', extensions: ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'mp4', 'mkv', 'mov', 'avi', 'h264', 'h265', 'hevc', 'raw', 'rgb', 'bgr', 'yuv'] }] });
  if (result.success) setInput(result.path);
}

function setInput(inputPath) {
  state.inputPath = inputPath;
  const name = inputPath.split(/[\\/]/).pop();
  const ext = name.includes('.') ? name.split('.').pop().toUpperCase() : '目录';
  $('input-kind').textContent = ext;
  $('selected-input').textContent = inputPath;
  $('selected-input').classList.remove('hidden');
  probeInput(inputPath);
  $('run-button').disabled = false;
}

async function probeInput(inputPath) {
  const result = await window.electronAPI.probeMedia(inputPath);
  if (!result.success) {
    $('media-info').textContent = '';
    return;
  }
  const info = result.info;
  const fps = info.fps ? `${info.fps.toFixed(2)} FPS` : 'FPS unknown';
  const duration = info.duration ? `${info.duration.toFixed(2)} s` : 'duration unknown';
  const frames = info.frames ? `${info.frames} frames` : 'frame count estimated/unknown';
  const prefix = state.config ? Math.min(info.duration || Infinity, (state.config.maxFrames || 8) / (state.config.sampleFps || 2)) : 0;
  $('media-info').textContent = `视频信息：${frames} · ${duration} · ${fps} · codec ${info.codec || 'unknown'}${prefix ? ` · 本次前缀窗口约 ${prefix.toFixed(2)} s` : ''}`;
}

function bandwidthValue() {
  const selected = $('bandwidth-preset').value;
  return selected === 'custom' ? Number($('bandwidth-custom').value) : Number(selected);
}

function appendResult(row) {
  const container = $('results');
  if (state.streamedResults === 0) container.innerHTML = '';
  state.streamedResults += 1;

  const item = document.createElement('article');
  item.className = 'result-row';
  const head = document.createElement('div'); head.className = 'result-head';
  const label = document.createElement('div'); label.className = 'result-label'; label.textContent = row.label || '结果';
  head.append(label); item.append(head);

  const response = document.createElement('div'); response.className = row.error ? 'response error' : 'response';
  response.textContent = row.error ? `失败：${row.error}` : (row.response || '（无响应）'); item.append(response);
  if (!row.error) {
    const metrics = document.createElement('div'); metrics.className = 'metrics';
    [
      ['端侧编码', ms(row.edge_encode_ms)],
      ['云端处理', ms(row.cloud_process_ms)],
      ['网络开销', ms(row.network_overhead_ms)],
      ['模拟上传', ms(row.upload_delay_ms)],
      ['端到端（TTFT）', ms(row.end_to_end_ttft_ms)],
      ['Payload', `${kb(row.payload_bytes)} / ${row.payload_scope || '总量'}`],
    ].forEach(([name, value]) => {
      const box = document.createElement('div'); box.className = 'metric';
      const small = document.createElement('small'); small.textContent = name;
      const strong = document.createElement('b'); strong.textContent = value;
      box.append(small, strong); metrics.append(box);
    });
    item.append(metrics);
    const detail = document.createElement('div'); detail.className = 'muted';
    const sampling = row.sample_fps ? ` · 采样 ${row.sample_fps} FPS / 前缀约 ${Number(row.sampled_prefix_seconds || 0).toFixed(2)} s` : '';
    const codecSource = row.temporal_redundancy_acceleration
      ? ` · codec 源帧处理 ${row.codec_source_frames_processed}`
        + ` · 输出帧型 ${(row.codec_selected_frame_types || []).join('/') || '未知'}`
        + ` · 编码负载 ${Number(row.codec_processing_realtime_factor || 0).toFixed(2)}× realtime`
        + ` · 含解码 ${Number(row.codec_total_realtime_factor || 0).toFixed(2)}× realtime`
      : '';
    const nativeTokens = row.pure_qwen ? ` · Qwen视觉token ${row.native_visual_tokens || 0} · grid ${JSON.stringify(row.native_video_grid_thw || [])}` : '';
    detail.textContent = `云端 TTFT ${ms(row.cloud_ttft_ms)} · HTTP 往返（含完整生成）${ms(row.http_roundtrip_ms)} · 云端解码 ${ms(row.cloud_decode_ms)} · 请求体 ${kb(row.request_bytes)} · 单帧 payload ${kb(row.payload_per_frame_bytes)} · ${row.bandwidth_kb_s ? `${row.bandwidth_kb_s} KB/s` : '带宽不限'}${sampling}${codecSource}${nativeTokens} · 特征 ${JSON.stringify(row.feature_shape || [])}`;
    item.append(detail);
  }
  container.append(item);
}

async function warmupCloud() {
  $('run-status').textContent = '云端模型预热中（不计入端到端结果）…';
  const result = await window.electronAPI.warmupCloudServer({ host: state.config.serverAddress || state.config.serverHost, port: state.config.serverPort });
  if (!result.success) throw new Error(result.error || '云端预热失败');
}

async function runInference() {
  if (state.running || !state.inputPath) return;
  if (state.projects.some((project) => project !== 'baseline') && !state.config.edgeCheckpoint) {
    $('settings-modal').classList.remove('hidden'); $('settings-message').textContent = '请先设置端侧 checkpoint'; return;
  }
  if (!state.projects.length) { $('run-status').textContent = '请先通过“＋ 添加项目”添加至少一个项目'; return; }
  state.running = true; state.streamedResults = 0; $('results').innerHTML = '<div class="empty">等待第一条消融结果…</div>'; $('run-button').disabled = true;
  try {
    if (!((await window.electronAPI.checkCloudServerStatus()).running)) {
      const start = await window.electronAPI.startCloudServer({
        checkpoint: state.config.cloudCheckpoint, host: state.config.serverHost, port: state.config.serverPort,
        device: state.config.device, qwenPath: state.config.qwenPath, offlineMode: state.config.offlineMode, preloadQwen: state.config.preloadQwen !== false,
      });
      if (!start.success) throw new Error(start.error || '云端启动失败');
      if (!start.health || !start.health.qwen_loaded) await warmupCloud();
    } else {
      await warmupCloud();
    }
    const result = await window.electronAPI.runDemoInference({
      inputPath: state.inputPath, edgeCheckpoint: state.config.edgeCheckpoint, serverUrl: serverUrl(state.config), prompt: $('prompt').value,
      device: state.config.device, timeout: state.config.timeout, maxFrames: state.config.maxFrames, spatialLevel: state.config.spatialLevel,
      projects: state.projects,
      sampleFps: state.config.sampleFps,
      bandwidthEnabled: $('bandwidth-enabled').checked, bandwidthKbS: bandwidthValue(), rawWidth: state.config.rawWidth,
      rawHeight: state.config.rawHeight, rawFps: state.config.rawFps, rawFormat: state.config.rawFormat,
    });
    if (!result.success) throw new Error(result.error || '推理失败');
    if (state.streamedResults === 0) (result.result.results || []).forEach(appendResult);
    $('run-status').textContent = `完成 · 模型加载 ${ms(result.result.model_load_ms)}（已剔除预热）`;
  } catch (error) {
    $('run-status').textContent = `失败：${error.message}`;
  }
  state.running = false; $('run-button').disabled = !state.inputPath;
}

document.addEventListener('DOMContentLoaded', async () => {
  const loaded = await window.electronAPI.loadConfig(); state.config = loaded.config; loadSettingsForm(state.config); setServerStatus((await window.electronAPI.checkCloudServerStatus()).running);
  $('choose-input').addEventListener('click', chooseInput);
  $('input-file').addEventListener('change', (event) => { if (event.target.files[0]) setInput(event.target.files[0].path); });
  ['dragover', 'dragenter'].forEach((eventName) => $('dropzone').addEventListener(eventName, (event) => { event.preventDefault(); $('dropzone').classList.add('drag'); }));
  $('dropzone').addEventListener('dragleave', () => $('dropzone').classList.remove('drag'));
  $('dropzone').addEventListener('drop', (event) => { event.preventDefault(); $('dropzone').classList.remove('drag'); const file = event.dataTransfer.files[0]; if (file?.path) setInput(file.path); });
  $('run-button').addEventListener('click', runInference);
  $('add-project').addEventListener('click', () => { state.projects.push('baseline'); renderProjects(); });
  $('clear-results').addEventListener('click', () => { state.streamedResults = 0; $('results').innerHTML = '<div class="empty">运行后显示响应与时延指标。</div>'; });
  $('server-button').addEventListener('click', startOrStopServer);
  $('settings-button').addEventListener('click', () => { loadSettingsForm(state.config); $('settings-modal').classList.remove('hidden'); });
  $('close-settings').addEventListener('click', () => $('settings-modal').classList.add('hidden')); $('save-settings').addEventListener('click', saveSettings);
  $('bandwidth-preset').addEventListener('change', () => { $('bandwidth-custom').disabled = $('bandwidth-preset').value !== 'custom'; });
  window.electronAPI.onCloudServerStatus((data) => setServerStatus(data));
  window.electronAPI.onDemoClientResult((row) => appendResult(row));
  window.electronAPI.onDemoClientLog((data) => { if (data.type === 'error') $('run-status').textContent = data.message; });
});
