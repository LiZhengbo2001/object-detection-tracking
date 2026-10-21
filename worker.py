"""Run the existing detector and tracker in an isolated process."""
import json
import os
import sys
import time
import shutil
import subprocess
from pathlib import Path

job = Path(sys.argv[1]).resolve()
config = json.loads((job / 'config.json').read_text())
root = Path(config['algorithm_root']).resolve()
os.chdir(root)
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'tracker'))

import cv2
import numpy as np
import track
from detector_utils import load_ultralytics_detector

sys.argv = ['track', '--tracker', config['tracker'], '--kalman_format', 'byte',
            '--device', config['device'], '--conf_thresh', str(config['confidence']),
            '--conf_thresh_low', str(config['low_confidence']), '--init_thresh', str(config['init_confidence']),
            '--track_buffer', str(config['track_buffer'])]
args = track.get_args()
tracker = track.TRACKER_DICT[config['tracker']](args)
is_rfdetr = config['detector'] == 'rfdetr_large'
if is_rfdetr:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('RF-DETR 需要可用的 CUDA 设备。')
    from rfdetr_utils import load_rfdetr_detector
    model = load_rfdetr_detector(config['weights'], root / '.venvs/rfdetr/bin/python')
else:
    model = load_ultralytics_detector(config['detector'], config['weights'])
source_type = config.get('source_type', 'video')
input_path = job / 'input' / config['filename']
cap = cv2.VideoCapture(0 if source_type == 'camera' else str(input_path))
if not cap.isOpened():
    raise RuntimeError('无法打开摄像头或读取输入文件，请检查设备和格式。')
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0 or fps > 240:
    fps = 25
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = 0 if source_type == 'camera' else (1 if source_type == 'image' else int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
writer = None
if source_type != 'image':
    writer = cv2.VideoWriter(str(job / 'result.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError('无法创建输出视频。')
ids = set()
trail = {}
frame_number = 0
start = time.monotonic()

def publish_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False))
    temporary.replace(path)

try:
    with (job / 'tracks.txt').open('w') as output:
        while not (job / 'stop').exists():
            ok, frame = cap.read()
            if not ok:
                break
            frame_started = time.monotonic()
            frame_number += 1
            if is_rfdetr:
                frame_path = job / 'rfdetr_frame.jpg'
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError('无法写入 RF-DETR 输入帧。')
                detections = model.predict(frame_path, config['low_confidence'])
            else:
                prediction = model.predict(frame, conf=config['low_confidence'], iou=config.get('iou', 0.7),
                                           imgsz=config['image_size'], device=config['device'], verbose=False)[0]
                detections = prediction.boxes.data.cpu().numpy().copy()
            detections = detections[np.isin(detections[:, 5].astype(int), [0, 2, 7])]
            detections[:, 2:4] -= detections[:, :2]
            tracks = tracker.update(detections, frame, frame)
            current = []
            for obj in tracks:
                x, y, w, h = obj.tlwh
                if w * h < config['min_area']:
                    continue
                ident, cls, score = int(obj.track_id), int(obj.category), float(obj.score)
                ids.add(ident)
                output.write(f'{frame_number},{ident},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{score:.4f},{cls},-1,-1\n')
                current.append({'id': ident, 'class_id': cls, 'score': score, 'bbox': [round(float(x), 1), round(float(y), 1), round(float(x+w), 1), round(float(y+h), 1)]})
                color = ((37 * ident) % 200 + 55, (97 * ident) % 200 + 55, (17 * ident) % 200 + 55)
                points = trail.setdefault(ident, [])
                points.append((int(x+w/2), int(y+h/2)))
                if len(points) > 40: del points[:-40]
                if config.get('show_trails', True) and len(points) > 1:
                    cv2.polylines(frame, [np.asarray(points, np.int32)], False, color, 2)
                if config.get('show_boxes', True):
                    cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
                    label = {0: 'person', 2: 'car', 7: 'truck'}.get(cls, str(cls))
                    cv2.putText(frame, f'{label} #{ident}', (int(x), max(15, int(y)-6)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
            if writer is not None:
                writer.write(frame)
            else:
                cv2.imwrite(str(job / 'result.jpg'), frame)
            preview_temp = job / 'preview_temp.jpg'
            if cv2.imwrite(str(preview_temp), frame, [cv2.IMWRITE_JPEG_QUALITY, 85]):
                preview_temp.replace(job / 'preview.jpg')
            publish_json(job / 'live.json', {'frame': frame_number, 'objects': current, 'seconds': round(time.monotonic()-frame_started, 3)})
            publish_json(job / 'progress.json', {'frames': frame_number, 'total': total, 'tracks': len(ids), 'elapsed': round(time.monotonic()-start, 1)})
            if source_type == 'image': break
finally:
    cap.release()
    if writer is not None: writer.release()
    if is_rfdetr: model.close()
publish_json(job / 'progress.json', {'frames': frame_number, 'total': total, 'tracks': len(ids), 'elapsed': round(time.monotonic()-start, 1)})
if not frame_number:
    raise RuntimeError('输入没有可解码的帧或任务在开始前已停止。')

# Postprocessing is offline: preserve the original rows and render the final rows.
stages = config.get('stages', [])
if stages:
    from postprocess.track_interpolation import load_mot
    from trajectory_optimizer.apply_temporal_residual import apply_directory
    from trajectory_optimizer.score_inpaint import process_directory as apply_a
    from trajectory_optimizer.tracklet_reliability_pipeline import process_directory as apply_d
    shutil.copyfile(job / 'tracks.txt', job / 'raw_tracks.txt')
    source = job / 'postprocess_input'
    source.mkdir()
    shutil.copyfile(job / 'tracks.txt', source / 'input.txt')
    records = {}
    weights = root / 'trajectory_optimizer_checkpoints/kitti'
    for stage in ('b', 'c', 'a', 'd'):
        if stage not in stages:
            continue
        (job / 'progress.json').write_text(json.dumps({'frames': frame_number, 'total': total, 'tracks': len(ids), 'elapsed': round(time.monotonic()-start, 1), 'stage': stage}))
        target = job / ('postprocess_' + stage)
        if stage == 'b':
            records[stage] = apply_directory(source, target, ['input'], weights / 'temporal_bbox_car_rfdetr_large2026_seed42_giou01_20261009.pt', device='cpu')
        elif stage == 'c':
            from postprocess.appearance_link.link import process_directory as apply_c
            rows = load_mot(source / 'input.txt')
            cars = rows[rows[:, 7].astype(int) == 2]
            if len(cars) == 0:
                target.mkdir()
                shutil.copyfile(source / 'input.txt', target / 'input.txt')
                records[stage] = {'message': '没有汽车轨迹，跳过外观关联', 'input_rows': len(rows)}
            else:
                # Extract only frames used by the original endpoint feature exporter.
                needed = set()
                for ident in np.unique(cars[:, 1]):
                    tr = cars[cars[:, 1] == ident]
                    tr = tr[np.argsort(tr[:, 0], kind='stable')]
                    needed.update(int(x) for x in np.r_[tr[:3, 0], tr[-3:, 0]])
                frame_dir = job / 'endpoint_images' / 'input'
                frame_dir.mkdir(parents=True)
                capture = cv2.VideoCapture(str(job / 'input' / config['filename']))
                try:
                    for index in range(1, max(needed) + 1):
                        ok, frame = capture.read()
                        if not ok: raise RuntimeError('提取 C 模块端点图像时读取视频失败。')
                        if index in needed and not cv2.imwrite(str(frame_dir / f'{index-1:06d}.png'), frame):
                            raise RuntimeError('无法保存 C 模块端点图像。')
                finally:
                    capture.release()
                features = job / 'endpoint_features'
                subprocess.run([str(root / '.venvs/rfdetr/bin/python'), str(root / 'tracker/postprocess/appearance_link/export_rfdetr_features.py'), '--input_dir', str(source), '--image_root', str(frame_dir.parent), '--output_dir', str(features), '--checkpoint', str(root / 'weights/rfdetr/rf-detr-large-2026-hf-original.pth'), '--sequences', 'input'], check=True)
                calibration = json.loads((weights / 'rfdetr_appearance_link_calibration_trainval_20261012.json').read_text())
                records[stage] = apply_c(source, features, target, ['input'], calibration, classes=[2], minimum_gap=1, maximum_gap=30, spatial_distance=75.0, minimum_length=2, enforce_disjoint=False)
        elif stage == 'a':
            records[stage] = apply_a(source, target, ['input'], weights / 'rfdetr_large2026_tracklet_quality_final_train0000_0014_20261013.joblib', weights / 'rfdetr_large2026_masked_gap_final_train0000_0014_t070_20261013.pt', enforce_disjoint=False, device='cpu')
            # This module normalizes sequence names with zfill(4).
        else:
            records[stage] = apply_d(source, target, ['input'], root / 'tracker/config_files/tracklet_reliability_rfdetr_kitti_trim_only.yaml')
        source = target
    shutil.copyfile(source / 'input.txt', job / 'tracks.txt')
    (job / 'postprocess.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
    rows = load_mot(job / 'tracks.txt')
    by_frame = {}
    for row in rows:
        by_frame.setdefault(int(row[0]), []).append(row)
    cap = cv2.VideoCapture(str(job / 'input' / config['filename']))
    final_video = job / 'result_final.mp4'
    writer = cv2.VideoWriter(str(final_video), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not cap.isOpened() or not writer.isOpened():
        raise RuntimeError('无法重新渲染后处理结果。')
    try:
        frame_index = 0
        last_rendered = None
        while frame_index < frame_number:
            ok, frame = cap.read()
            if not ok: break
            frame_index += 1
            for row in by_frame.get(frame_index, []):
                _, ident, x, y, w, h, score, cls, _, _ = row
                ident, cls = int(ident), int(cls)
                color = ((37 * ident) % 200 + 55, (97 * ident) % 200 + 55, (17 * ident) % 200 + 55)
                cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
                cv2.putText(frame, f"{ {0:'person', 2:'car', 7:'truck'}.get(cls, str(cls))} #{ident}", (int(x), max(15, int(y)-6)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
            writer.write(frame)
            last_rendered = frame
    finally:
        cap.release()
        writer.release()
    final_video.replace(job / 'result.mp4')
    if frame_index:
        cv2.imwrite(str(job / 'preview_temp.jpg'), last_rendered, [cv2.IMWRITE_JPEG_QUALITY, 85])
        (job / 'preview_temp.jpg').replace(job / 'preview.jpg')
        final_objects = [{'id': int(r[1]), 'class_id': int(r[7]), 'score': float(r[6]), 'bbox': [round(float(r[2]), 1), round(float(r[3]), 1), round(float(r[2]+r[4]), 1), round(float(r[3]+r[5]), 1)]} for r in by_frame.get(frame_index, [])]
        publish_json(job / 'live.json', {'frame': frame_index, 'objects': final_objects, 'seconds': None, 'postprocessed': True})
    (job / 'progress.json').write_text(json.dumps({'frames': frame_number, 'total': total, 'tracks': len(set(rows[:, 1].astype(int))), 'elapsed': round(time.monotonic()-start, 1)}))
