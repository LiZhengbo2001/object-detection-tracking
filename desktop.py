"""Native desktop workbench for the local tracking service."""
import http.client
import json
import os
import shutil
import socket
import sys
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from PyQt5.QtCore import Qt, QProcess, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QScrollArea, QSpinBox, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QHeaderView)

ROOT = Path(__file__).resolve().parent
NAMES = {0: '行人', 2: '汽车', 7: '卡车'}
STATES = {'queued': '排队中', 'running': '检测跟踪中', 'completed': '处理完成',
          'failed': '处理失败', 'interrupted': '任务中断'}

class Upload(QThread):
    created = pyqtSignal(object)
    progress = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, port, paths, options):
        super().__init__()
        self.port, self.paths, self.options = port, paths, options

    def run(self):
        try:
            for i, path in enumerate(self.paths):
                self.progress.emit(f'正在提交 {i + 1} / {len(self.paths)}')
                conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=300)
                size = path.stat().st_size if path else 0
                conn.putrequest('POST', '/api/jobs')
                conn.putheader('Content-Length', str(size))
                conn.putheader('Content-Type', 'application/octet-stream')
                conn.putheader('X-Filename', quote(path.name if path else ''))
                conn.putheader('X-Options', quote(json.dumps(self.options)))
                conn.endheaders()
                if path:
                    with path.open('rb') as stream:
                        while chunk := stream.read(1024 * 1024): conn.send(chunk)
                response = conn.getresponse()
                body = json.loads(response.read())
                if response.status >= 400: raise RuntimeError(body.get('error', '提交失败'))
                self.created.emit(body)
                conn.close()
        except Exception as exc:
            self.error.emit(str(exc))

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('交通目标检测与跟踪系统')
        self.resize(1420, 920)
        self.selected = None
        self.last_frame = None
        self.last_state = None
        self.current_objects = []
        self.upload = None
        self.latest_pixmap = QPixmap()
        self.service = QProcess(self)
        sock = socket.socket()
        sock.bind(('127.0.0.1', 0))
        self.port = sock.getsockname()[1]
        sock.close()
        env = self.service.processEnvironment()
        from PyQt5.QtCore import QProcessEnvironment
        env = QProcessEnvironment.systemEnvironment()
        env.insert('PORT', str(self.port))
        self.service.setProcessEnvironment(env)
        self.service.setWorkingDirectory(str(ROOT))
        self.service.start(sys.executable, [str(ROOT / 'server.py')])
        self.make_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(500)
        QTimer.singleShot(1000, self.load_config)

    def get(self, path):
        with urlopen(f'http://127.0.0.1:{self.port}{path}', timeout=2) as response:
            return json.load(response)

    def make_ui(self):
        container = QWidget()
        self.setCentralWidget(container)
        main = QVBoxLayout(container)
        main.setContentsMargins(22, 15, 22, 15)
        title = QLabel('◈   交通目标检测与跟踪系统')
        title.setObjectName('title')
        main.addWidget(title)
        subtitle = QLabel('交通智能感知  /  人员与车辆检测、关联及轨迹分析')
        subtitle.setObjectName('subtitle')
        main.addWidget(subtitle)
        split = QSplitter(Qt.Horizontal)
        main.addWidget(split, 1)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        preview_box = QGroupBox('  画面预览  /  LIVE PREVIEW')
        pv = QVBoxLayout(preview_box)
        self.preview = QLabel('等待画面输入\n\n打开图片、视频或本机摄像头')
        self.preview.setObjectName('preview')
        self.preview.setMinimumSize(480, 320)
        self.preview.setAlignment(Qt.AlignCenter)
        pv.addWidget(self.preview, 1)
        self.status = QLabel('就绪  ·  请选择输入源')
        pv.addWidget(self.status)
        left_layout.addWidget(preview_box, 3)
        results = QGroupBox('  当前帧检测结果与位置')
        table_layout = QVBoxLayout(results)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(['序号', '跟踪 ID', '类别', '置信度', '位置 (xmin, ymin, xmax, ymax)'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.itemSelectionChanged.connect(self.table_selected)
        table_layout.addWidget(self.table)
        left_layout.addWidget(results, 1)
        split.addWidget(left)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(355)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 0, 0, 0)
        settings = QGroupBox('  参数设置')
        form = QFormLayout(settings)
        toggles = QHBoxLayout()
        self.boxes = QCheckBox('显示检测框'); self.boxes.setChecked(True)
        self.trails = QCheckBox('显示跟踪轨迹'); self.trails.setChecked(True)
        toggles.addWidget(self.boxes); toggles.addWidget(self.trails)
        form.addRow(toggles)
        self.weights = QComboBox(); form.addRow('检测模型', self.weights)
        self.tracker = QComboBox(); self.tracker.addItem('ByteSort', 'traffic_bytetrack'); self.tracker.addItem('ByteTrack', 'bytetrack'); form.addRow('跟踪算法', self.tracker)
        self.device = QComboBox(); self.device.addItem('CPU', 'cpu'); self.device.addItem('GPU 0', '0'); form.addRow('运行设备', self.device)
        self.conf = self.number(.5); self.iou = self.number(.7)
        self.low = self.number(.1); self.init = self.number(.6)
        form.addRow('置信度阈值', self.conf); form.addRow('交并比阈值', self.iou)
        form.addRow('低分检测阈值', self.low); form.addRow('初始化阈值', self.init)
        self.size = QSpinBox(); self.size.setRange(320, 1920); self.size.setSingleStep(32); self.size.setValue(640); form.addRow('输入尺寸', self.size)
        self.stage_boxes = {}
        stages = QGridLayout()
        for index, (key, text) in enumerate([('b', 'B 时序优化'), ('c', 'C 轨迹连接'), ('a', 'A 缺口补全'), ('d', 'D 边界控制')]):
            self.stage_boxes[key] = QCheckBox(text); stages.addWidget(self.stage_boxes[key], index // 2, index % 2)
        form.addRow('视频后处理', stages)
        note = QLabel('后处理采用 KITTI/RF-DETR 权重，C 需要 GPU。设置在任务开始时生效。')
        note.setWordWrap(True); note.setObjectName('muted'); form.addRow(note)
        right_layout.addWidget(settings)
        stats = QGroupBox('  检测结果')
        grid = QGridLayout(stats)
        self.stats = {}
        for index, (key, text) in enumerate([('all', '目标总数'), ('time', '单帧用时'), ('person', '行人数量'), ('car', '车辆数量')]):
            cell = QLabel(text + '\n—'); cell.setObjectName('metric'); self.stats[key] = cell; grid.addWidget(cell, index // 2, index % 2)
        self.target = QComboBox(); self.target.addItem('请选择目标', None); self.target.currentIndexChanged.connect(self.target_selected)
        grid.addWidget(QLabel('目标选择'), 2, 0); grid.addWidget(self.target, 2, 1)
        self.detail = QLabel('类型  —       置信度  —\n\nxmin  —       ymin  —\n\nxmax  —       ymax  —')
        self.detail.setObjectName('details'); grid.addWidget(self.detail, 3, 0, 1, 2)
        right_layout.addWidget(stats)
        operations = QGroupBox('  操作面板')
        op = QGridLayout(operations)
        buttons = [('打开图片', self.open_image), ('打开文件夹', self.open_folder), ('打开视频', self.open_video), ('打开摄像头', self.open_camera), ('停止任务', self.stop), ('保存画面', self.save_result), ('保存轨迹 TXT', self.save_tracks), ('查看运行日志', self.show_log)]
        for index, (text, fn) in enumerate(buttons):
            button = QPushButton(text); button.clicked.connect(fn); op.addWidget(button, index // 2, index % 2)
        right_layout.insertWidget(0, operations)
        history = QGroupBox('  任务记录')
        hist = QVBoxLayout(history)
        self.history = QComboBox(); self.history.activated.connect(self.select_history); hist.addWidget(self.history)
        reload_button = QPushButton('刷新任务记录'); reload_button.clicked.connect(self.reload_history); hist.addWidget(reload_button)
        right_layout.addWidget(history)
        right_layout.addStretch()
        scroll.setWidget(right); split.addWidget(scroll); split.setStretchFactor(0, 3); split.setStretchFactor(1, 1)

    @staticmethod
    def number(value):
        widget = QDoubleSpinBox(); widget.setRange(0, 1); widget.setSingleStep(.05); widget.setDecimals(2); widget.setValue(value); return widget

    def load_config(self):
        try:
            self.weights.clear(); self.weights.addItems(self.get('/api/config')['weights'])
            self.reload_history()
        except Exception as exc: self.status.setText(f'服务启动失败：{exc}')

    def options(self, source):
        return {'source_type': source, 'weights': self.weights.currentText(), 'tracker': self.tracker.currentData(),
                'device': self.device.currentData(), 'confidence': self.conf.value(), 'iou': self.iou.value(),
                'low_confidence': self.low.value(), 'init_confidence': self.init.value(), 'image_size': self.size.value(),
                'show_boxes': self.boxes.isChecked(), 'show_trails': self.trails.isChecked(),
                'stages': [key for key, box in self.stage_boxes.items() if box.isChecked()]}

    def submit(self, paths, source):
        if self.upload and self.upload.isRunning():
            QMessageBox.information(self, '请稍候', '正在提交文件。'); return
        self.upload = Upload(self.port, paths, self.options(source))
        self.upload.created.connect(self.created); self.upload.progress.connect(self.status.setText)
        self.upload.error.connect(lambda text: QMessageBox.warning(self, '任务提交失败', text))
        self.upload.finished.connect(self.reload_history); self.upload.start()

    def created(self, job):
        # A folder is a sequence of independent image jobs; show the first.
        if self.upload and len(self.upload.paths) > 1 and self.upload.paths[0].name != job['config']['filename']: return
        self.selected = job['id']; self.last_frame = None; self.last_state = None
        self.status.setText('任务已提交')

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, '打开图片', '', '图片 (*.jpg *.jpeg *.png *.bmp *.webp)')
        if path: self.submit([Path(path)], 'image')
    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, '打开视频', '', '视频 (*.mp4 *.avi *.mov *.mkv)')
        if path: self.submit([Path(path)], 'video')
    def open_folder(self):
        path = QFileDialog.getExistingDirectory(self, '打开图片文件夹')
        if path:
            files = sorted(p for p in Path(path).iterdir() if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
            if files: self.submit(files, 'image')
            else: QMessageBox.information(self, '没有图片', '该文件夹中没有支持的图片。')
    def open_camera(self): self.submit([None], 'camera')
    def stop(self):
        if not self.selected: return
        try:
            conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=2)
            conn.request('POST', f'/api/jobs/{self.selected}/stop', body=b'')
            response = conn.getresponse(); data = json.loads(response.read()); conn.close()
            if response.status >= 400: raise RuntimeError(data.get('error', '停止失败'))
            self.status.setText('正在停止任务…')
        except Exception as exc: QMessageBox.warning(self, '停止失败', str(exc))

    def refresh(self):
        if not self.selected: return
        try:
            job = self.get('/api/jobs/' + self.selected)
            live = job.get('live', {}); p = job.get('progress', {}); state = job['status']['state']
            stage = p.get('stage')
            self.status.setText(f"{job['config']['filename']}  ·  {('后处理 ' + stage.upper()) if stage else STATES.get(state, state)}  ·  帧 {p.get('frames', 0)}  ·  累计轨迹 {p.get('tracks', 0)}")
            if state == 'failed' and self.last_state != 'failed':
                QMessageBox.warning(self, '任务失败', job['status'].get('message', '请查看运行日志。'))
            frame = live.get('frame')
            if frame and (frame != self.last_frame or state != self.last_state):
                with urlopen(f'http://127.0.0.1:{self.port}/api/jobs/{self.selected}/preview.jpg?frame={frame}', timeout=2) as response:
                    self.latest_pixmap.loadFromData(response.read())
                self.update_preview(); self.update_objects(live)
                self.last_frame = frame
            if self.last_state != state and state in ('completed', 'failed', 'interrupted'): self.reload_history()
            self.last_state = state
        except Exception as exc: self.status.setText('服务连接失败：' + str(exc))

    def update_preview(self):
        if not self.latest_pixmap.isNull(): self.preview.setPixmap(self.latest_pixmap.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'preview'): self.update_preview()

    def update_objects(self, live):
        self.current_objects = live.get('objects', [])
        objs = self.current_objects
        self.stats['all'].setText(f'目标总数\n{len(objs)} 个')
        self.stats['person'].setText(f"行人数量\n{sum(o['class_id'] == 0 for o in objs)} 个")
        self.stats['car'].setText(f"车辆数量\n{sum(o['class_id'] in (2, 7) for o in objs)} 个")
        sec = live.get('seconds'); self.stats['time'].setText(f"单帧用时\n{f'{sec:.3f} 秒' if sec is not None else '—'}")
        selected_id = self.target.currentData()
        self.target.blockSignals(True); self.target.clear(); self.target.addItem('请选择目标', None)
        self.table.setRowCount(len(objs))
        for i, obj in enumerate(objs):
            self.target.addItem(f"#{obj['id']} · {NAMES.get(obj['class_id'], obj['class_id'])}", obj['id'])
            values = [str(i+1), str(obj['id']), NAMES.get(obj['class_id'], str(obj['class_id'])), f"{obj['score']*100:.2f}%", ', '.join(map(str, obj['bbox']))]
            for column, value in enumerate(values): self.table.setItem(i, column, QTableWidgetItem(value))
        idx = self.target.findData(selected_id); self.target.setCurrentIndex(max(0, idx)); self.target.blockSignals(False); self.target_selected()
    def table_selected(self):
        row = self.table.currentRow()
        if 0 <= row < len(self.current_objects): self.target.setCurrentIndex(self.target.findData(self.current_objects[row]['id']))
    def target_selected(self):
        obj = next((o for o in self.current_objects if o['id'] == self.target.currentData()), None)
        if not obj: self.detail.setText('类型  —       置信度  —\n\nxmin  —       ymin  —\n\nxmax  —       ymax  —'); return
        x1,y1,x2,y2 = obj['bbox']; self.detail.setText(f"类型  {NAMES.get(obj['class_id'])}       置信度  {obj['score']*100:.2f}%\n\nxmin  {x1}       ymin  {y1}\n\nxmax  {x2}       ymax  {y2}")
    def reload_history(self):
        try:
            jobs = sorted(self.get('/api/jobs'), key=lambda j: j['config'].get('created', 0), reverse=True)
            self.history.blockSignals(True); self.history.clear()
            for job in jobs[:100]: self.history.addItem(f"{job['config']['filename']}  ·  {STATES.get(job['status']['state'], job['status']['state'])}", job['id'])
            index = self.history.findData(self.selected)
            if index >= 0: self.history.setCurrentIndex(index)
            self.history.blockSignals(False)
        except Exception: pass
    def select_history(self, index): self.selected = self.history.itemData(index); self.last_frame = None; self.last_state = None
    def save_result(self):
        if not self.selected: return
        job = self.get('/api/jobs/' + self.selected)
        if job['status']['state'] != 'completed': QMessageBox.information(self, '尚未完成', '任务完成后才能保存。'); return
        suffix = '.jpg' if job['config'].get('source_type') == 'image' else '.mp4'
        self.save_file('result' + suffix, '保存结果画面')
    def save_tracks(self): self.save_file('tracks.txt', '保存轨迹文件')
    def save_file(self, name, caption):
        if not self.selected: return
        source = ROOT / 'runtime/jobs' / self.selected / name
        if not source.is_file(): QMessageBox.information(self, '没有结果', '结果文件尚未生成。'); return
        path, _ = QFileDialog.getSaveFileName(self, caption, name)
        if path:
            try: shutil.copyfile(source, path)
            except OSError as exc: QMessageBox.warning(self, '保存失败', str(exc))
    def show_log(self):
        if self.selected:
            from PyQt5.QtCore import QUrl
            from PyQt5.QtGui import QDesktopServices
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(ROOT / 'runtime/jobs' / self.selected / 'run.log')))
    def closeEvent(self, event):
        if self.upload and self.upload.isRunning():
            QMessageBox.information(self, '请稍候', '请等待文件提交完成再关闭。'); event.ignore(); return
        try:
            jobs = self.get('/api/jobs')
            active_jobs = [j for j in jobs if j['status']['state'] in ('queued', 'running') and j['status'].get('server_pid') == self.service.processId()]
            if active_jobs:
                answer = QMessageBox.question(self, '结束任务', f'关闭窗口将中断 {len(active_jobs)} 个正在运行或排队的任务，是否继续？')
                if answer != QMessageBox.Yes: event.ignore(); return
        except Exception: pass
        self.service.terminate(); self.service.waitForFinished(2000)
        event.accept()

STYLE = '''
QWidget {font-family: "Microsoft YaHei", "Noto Sans CJK SC", sans-serif; font-size: 13px; color: #26433e; background: #f2f6f5;}
QLabel#title {font-size: 22px; font-weight: bold; color: #153f36; padding: 4px;}
QLabel#subtitle, QLabel#muted {color: #829b95; font-size: 11px;}
QGroupBox {background: #fff; border: 1px solid #dce9e4; border-radius: 8px; margin-top: 12px; padding: 15px 10px 10px; font-weight: bold;}
QGroupBox::title {subcontrol-origin: margin; left: 10px; color: #29896c;}
QLabel#preview {background: #102d31; color: #9abcb3; border-radius: 5px; font-size: 15px;}
QLabel#metric {background: #f0f8f4; color: #24866b; padding: 11px; border-radius: 5px; font-size: 13px;}
QLabel#details {background: #f7faf9; color: #4a776b; padding: 10px;}
QPushButton {background: #eaf7f0; color: #23795c; border: 1px solid #cfe9db; border-radius: 5px; padding: 9px;}
QPushButton:hover {background: #d7f1e2;}
QComboBox, QSpinBox, QDoubleSpinBox {background: white; border: 1px solid #d8e7e1; border-radius: 4px; padding: 4px; min-height: 24px;}
QTableWidget {background: white; gridline-color: #edf3f0; border: 0;}
QHeaderView::section {background: #f3f8f6; padding: 8px; border: 0; color: #76938a;}
'''

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    window = Window(); window.show()
    sys.exit(app.exec_())
