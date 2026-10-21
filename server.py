"""Local single-user tracking platform; model execution stays in the original project."""
import json
import mimetypes
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
JOBS = ROOT / 'runtime' / 'jobs'
ALGORITHM = Path(os.environ.get('TRACKING_ALGORITHM_ROOT', '/home/lzb/Yolov7-tracker-2.1 (2)')).resolve()
PYTHON = os.environ.get('TRACKING_PYTHON', sys.executable)
JOBS.mkdir(parents=True, exist_ok=True)
lock = threading.Lock()
active = set()
processes = {}

def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default

def job_info(path):
    config = read_json(path / 'config.json', {})
    status = read_json(path / 'status.json', {'state': 'interrupted', 'message': '服务已重启，任务中断'})
    if status.get('state') in ('running', 'queued') and path.name not in active:
        owner = status.get('server_pid')
        try:
            if not owner: raise ProcessLookupError()
            os.kill(owner, 0)
        except (ProcessLookupError, PermissionError):
            status = {'state': 'interrupted', 'message': '服务已重启，任务中断'}
    return {'id': path.name, 'config': {k: v for k, v in config.items() if k not in ('algorithm_root',)},
            'status': status, 'progress': read_json(path / 'progress.json', {}),
            'result': (path / 'result.mp4').exists() or (path / 'result.jpg').exists(), 'tracks': (path / 'tracks.txt').exists(),
            'live': read_json(path / 'live.json', {})}

def run_job(path):
    with lock:
        try:
            (path / 'status.json').write_text(json.dumps({'state': 'running', 'server_pid': os.getpid()}))
            with (path / 'run.log').open('w') as log:
                proc = subprocess.Popen([PYTHON, str(ROOT / 'worker.py'), str(path)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                processes[path.name] = proc
                returncode = proc.wait()
                processes.pop(path.name, None)
            state = 'completed' if returncode == 0 else 'failed'
            (path / 'status.json').write_text(json.dumps({'state': state, 'message': '' if not returncode else '处理失败，请查看运行日志。'}))
        except Exception as exc:
            (path / 'status.json').write_text(json.dumps({'state': 'failed', 'message': str(exc)}))
        finally:
            active.discard(path.name)

class Handler(BaseHTTPRequestHandler):
    def send_json(self, data, code=200):
        payload = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code); self.send_header('Content-Type', 'application/json; charset=utf-8'); self.send_header('Content-Length', str(len(payload))); self.end_headers(); self.wfile.write(payload)
    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/api/config':
            weights = [str(p.relative_to(ALGORITHM)) for p in sorted((ALGORITHM / 'weights').glob('*.pt'))] if ALGORITHM.exists() else []
            if (ALGORITHM / 'yolov8n.pt').exists(): weights.insert(0, 'yolov8n.pt')
            rfdetr = ALGORITHM / 'weights/rfdetr/rf-detr-large-2026-hf-original.pth'
            if rfdetr.exists() and (ALGORITHM / '.venvs/rfdetr/bin/python').exists(): weights.append(str(rfdetr.relative_to(ALGORITHM)))
            return self.send_json({'weights': weights, 'algorithm_available': (ALGORITHM / 'tracker/track.py').exists()})
        if path == '/api/jobs':
            return self.send_json([job_info(p) for p in sorted(JOBS.iterdir(), reverse=True) if p.is_dir() and (p / 'config.json').exists()])
        match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})(?:/(result.mp4|result.jpg|preview.jpg|tracks.txt|raw_tracks.txt|postprocess.json|run.log))?', path)
        if match:
            job = JOBS / match[1]
            if not job.is_dir(): return self.send_json({'error': '任务不存在'}, 404)
            if not match[2]: return self.send_json(job_info(job))
            return self.send_file(job / match[2])
        if path == '/': path = '/index.html'
        if path in ('/index.html', '/style.css', '/app.js'): return self.send_file(ROOT / 'web' / path[1:])
        return self.send_json({'error': '未找到'}, 404)
    def send_file(self, path):
        if not path.is_file(): return self.send_json({'error': '文件不存在'}, 404)
        size = path.stat().st_size
        self.send_response(200); self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream'); self.send_header('Content-Length', str(size)); self.end_headers()
        with path.open('rb') as f:
            while chunk := f.read(1024 * 1024): self.wfile.write(chunk)
    def do_POST(self):
        path = urlparse(self.path).path
        stop = re.fullmatch(r'/api/jobs/([a-f0-9]{32})/stop', path)
        if stop:
            job = JOBS / stop[1]
            if not job.is_dir() or stop[1] not in active: return self.send_json({'error': '任务未运行'}, 400)
            (job / 'stop').touch()
            return self.send_json({'ok': True})
        if path != '/api/jobs': return self.send_json({'error': '未找到'}, 404)
        try:
            filename = Path(unquote(self.headers.get('X-Filename', ''))).name
            options = json.loads(unquote(self.headers.get('X-Options', '{}')))
            source_type = options.get('source_type', 'video')
            if source_type not in ('video', 'image', 'camera'): raise ValueError('不支持的输入类型。')
            suffixes = ('.mp4', '.avi', '.mov', '.mkv') if source_type == 'video' else ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
            if source_type != 'camera' and (not filename or Path(filename).suffix.lower() not in suffixes): raise ValueError('文件类型与选择的输入不匹配。')
            if source_type == 'camera': filename = '摄像头 0'
            length = int(self.headers.get('Content-Length', '0'))
            if (source_type != 'camera' and length <= 0) or length > 2 * 1024**3: raise ValueError('文件为空或超过 2 GB。')
            tracker = options.get('tracker', 'traffic_bytetrack')
            if tracker not in ('traffic_bytetrack', 'bytetrack'): raise ValueError('不支持的跟踪器。')
            weight = str(options.get('weights', 'yolov8n.pt'))
            weight_path = (ALGORITHM / weight).resolve()
            if not weight_path.is_relative_to(ALGORITHM) or not weight_path.is_file() or weight_path.suffix not in ('.pt', '.pth'): raise ValueError('权重文件不存在。')
            confidence, low, init = (float(options.get(k, d)) for k, d in [('confidence', .5), ('low_confidence', .1), ('init_confidence', .6)])
            if not 0 <= low <= confidence <= 1 or not confidence <= init <= 1: raise ValueError('阈值须满足 0 ≤ 低阈值 ≤ 跟踪阈值 ≤ 初始化阈值 ≤ 1。')
            device = options.get('device', 'cpu')
            if device not in ('cpu', '0'): raise ValueError('设备只能选择 CPU 或 GPU 0。')
            image_size, track_buffer, min_area = int(options.get('image_size', 640)), int(options.get('track_buffer', 30)), float(options.get('min_area', 150))
            if not 320 <= image_size <= 1920 or not 1 <= track_buffer <= 300 or not 0 <= min_area <= 1000000: raise ValueError('参数超出允许范围。')
            detector = ('rfdetr_large' if weight_path.suffix == '.pth' else 'yolo_ultra_rtdetr' if weight_path.name.lower().startswith('rtdetr') else 'yolo_ultra_v8')
            if detector == 'rfdetr_large' and (weight != 'weights/rfdetr/rf-detr-large-2026-hf-original.pth' or device != '0'): raise ValueError('RF-DETR 需要指定项目内官方权重和 GPU 0。')
            stages = options.get('stages', [])
            if not isinstance(stages, list) or any(stage not in ('b', 'c', 'a', 'd') for stage in stages) or len(set(stages)) != len(stages): raise ValueError('后处理配置无效。')
            if stages and source_type != 'video': raise ValueError('A/B/C/D 离线后处理仅支持视频任务。')
            if 'c' in stages and device != '0': raise ValueError('C 模块需要 GPU 0 和 RF-DETR 独立环境。')
            job = JOBS / uuid.uuid4().hex; (job / 'input').mkdir(parents=True)
            with (job / 'input' / filename).open('wb') as f:
                remaining = length
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk: raise ValueError('上传中断。')
                    f.write(chunk); remaining -= len(chunk)
            iou = float(options.get('iou', 0.7))
            if not 0 <= iou <= 1: raise ValueError('交并比阈值必须在 0 到 1 之间。')
            config = {'source_type': source_type, 'show_boxes': bool(options.get('show_boxes', True)), 'show_trails': bool(options.get('show_trails', True)), 'iou': iou, 'algorithm_root': str(ALGORITHM), 'filename': filename, 'weights': str(weight_path), 'tracker': tracker, 'detector': detector, 'device': device, 'confidence': confidence, 'low_confidence': low, 'init_confidence': init, 'image_size': image_size, 'track_buffer': track_buffer, 'min_area': min_area, 'stages': stages, 'created': time.time()}
            (job / 'config.json').write_text(json.dumps(config, ensure_ascii=False)); (job / 'status.json').write_text(json.dumps({'state': 'queued', 'server_pid': os.getpid()}))
            active.add(job.name); threading.Thread(target=run_job, args=(job,), daemon=True).start()
            self.send_json(job_info(job), 201)
        except (ValueError, TypeError, OSError, json.JSONDecodeError) as exc:
            self.send_json({'error': str(exc)}, 400)

def terminate_children(signum, frame):
    for proc in list(processes.values()):
        try: os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError: pass
    raise SystemExit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, terminate_children)
    port = int(os.environ.get('PORT', '8000'))
    print(f'平台地址：http://127.0.0.1:{port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()
