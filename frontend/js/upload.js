export const MAX_UPLOAD_MB = 10;
export const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024;
export function validateFile(file, maxMB = MAX_UPLOAD_MB) {
  if (!file) return 'Choose a file first.';
  if (!/\.(png|jpe?g|svg|pdf)$/i.test(file.name)) return 'Use a PNG, JPG, SVG, or single-page PDF file.';
  if (!file.size) return 'This file is empty. Export the diagram again.';
  if (file.size > maxMB * 1024 * 1024) return `This file is too large. Choose a file no larger than ${maxMB} MB.`;
  return null;
}
export function bindUpload(onSelect) {
  const input = document.getElementById('file-input');
  const panel = document.getElementById('diagram-panel');
  const dropZone = document.getElementById('diagram-empty');
  const choose = document.getElementById('choose-button');
  let dragDepth = 0;

  choose.addEventListener('click', () => input.click());
  input.addEventListener('change', () => {
    if (input.files[0]) onSelect(input.files[0]);
  });

  dropZone.addEventListener('dragenter', event => {
    event.preventDefault();
    dragDepth += 1;
    dropZone.classList.add('dragging');
  });
  dropZone.addEventListener('dragover', event => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
    dropZone.classList.add('dragging');
  });
  dropZone.addEventListener('dragleave', event => {
    event.preventDefault();
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) dropZone.classList.remove('dragging');
  });
  dropZone.addEventListener('drop', event => {
    event.preventDefault();
    dragDepth = 0;
    dropZone.classList.remove('dragging');
    if (event.dataTransfer.files.length !== 1) {
      document.getElementById('file-error').textContent = 'Choose exactly one flowchart at a time.';
      return;
    }
    onSelect(event.dataTransfer.files[0]);
  });
}
export function upload(file, onProgress) {
  const request = new XMLHttpRequest();
  const promise = new Promise((resolve, reject) => {
    request.open('POST', '/api/analyze'); request.responseType = 'json'; request.timeout = 180000;
    request.upload.addEventListener('progress', event => { if (event.lengthComputable) onProgress(Math.round(event.loaded / event.total * 100)); });
    request.upload.addEventListener('load', () => onProgress(null));
    request.addEventListener('load', () => {
      if (request.status >= 200 && request.status < 300 && request.response) resolve(request.response);
      else reject(new Error(request.response?.message || 'The server could not analyze this file. Try again.'));
    });
    request.addEventListener('error', () => reject(new Error('Cannot reach the local server. Check that Flask is running, then try again.')));
    request.addEventListener('timeout', () => reject(new Error('Analysis took too long. Try a smaller, clearer export.')));
    request.addEventListener('abort', () => reject(new DOMException('Request cancelled', 'AbortError')));
    const form = new FormData(); form.append('file', file); request.send(form);
  });
  return { promise, cancel: () => request.abort() };
}
