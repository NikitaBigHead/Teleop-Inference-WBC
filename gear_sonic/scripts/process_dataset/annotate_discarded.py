#!/usr/bin/env python3
"""Review episode videos and mark discarded episodes in meta/info.json."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
import tempfile


DEFAULT_DATASET = Path(
    "/home/nikita/Skoltech/MWS/Teleop-Data-Collection/outputs/extended-sep-24-with-none"
)
PREFERRED_CAMERA = "observation.images.external_view_camera"


def episode_indices(path: Path) -> tuple[list[int], dict[int, str]]:
    indices: list[int] = []
    descriptions: dict[int, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                index = record["episode_index"]
                if type(index) is not int or index < 0:
                    raise ValueError("episode_index must be a non-negative integer")
                if index in descriptions:
                    raise ValueError(f"duplicate episode_index {index}")
                tasks = record.get("tasks", [])
                descriptions[index] = ", ".join(map(str, tasks)) if isinstance(tasks, list) else ""
                indices.append(index)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}, line {line_number}: {exc}") from exc
    if not indices:
        raise ValueError(f"No episodes in {path}")
    return sorted(indices), descriptions


def read_discarded(info: dict, known: set[int]) -> set[int]:
    raw = info.get("discarded_episode_indices", [])
    if not isinstance(raw, list):
        raise ValueError("discarded_episode_indices must be a JSON array")
    if any(type(index) is not int or index not in known for index in raw):
        raise ValueError("discarded_episode_indices contains an invalid or unknown episode ID")
    return set(raw)


class Dataset:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.info_path = self.root / "meta/info.json"
        self.episodes, self.descriptions = episode_indices(self.root / "meta/episodes.jsonl")
        self.known = set(self.episodes)
        self.discarded = self.reload_discarded()
        self.videos: dict[str, dict[int, Path]] = {}
        for path in sorted((self.root / "videos").glob("chunk-*/*/episode_*.mp4")):
            match = re.fullmatch(r"episode_(\d+)\.mp4", path.name)
            if match is None:
                continue
            index = int(match[1])
            if index not in self.known:
                continue
            camera_videos = self.videos.setdefault(path.parent.name, {})
            if index in camera_videos:
                raise ValueError(f"Duplicate {path.parent.name} video for episode {index}")
            camera_videos[index] = path
        if not self.videos:
            raise ValueError(f"No episode videos found in {self.root / 'videos'}")
        self.cameras = sorted(self.videos, key=lambda name: (name != PREFERRED_CAMERA, name))

    def reload_discarded(self) -> set[int]:
        info = json.loads(self.info_path.read_text(encoding="utf-8"))
        if not isinstance(info, dict):
            raise ValueError(f"{self.info_path} must contain a JSON object")
        self.discarded = read_discarded(info, self.known)
        return self.discarded

    def set_discarded(self, episode: int, discarded: bool) -> set[int]:
        if episode not in self.known:
            raise ValueError(f"Unknown episode {episode}")
        path = self.info_path
        with path.with_name("info.json.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            original = path.read_bytes()
            info = json.loads(original)
            if not isinstance(info, dict):
                raise ValueError(f"{path} must contain a JSON object")
            original_discarded = read_discarded(info, self.known)
            updated = original_discarded.copy()
            if discarded:
                updated.add(episode)
            else:
                updated.discard(episode)
            if updated == original_discarded and "discarded_episode_indices" in info:
                self.discarded = updated
                return updated
            info["discarded_episode_indices"] = sorted(updated)
            content = (json.dumps(info, ensure_ascii=False, indent=4, allow_nan=False) + "\n").encode("utf-8")
            backup = path.with_name("info.json.bak")
            try:
                with backup.open("xb") as stream:
                    stream.write(original)
                    stream.flush()
                    os.fsync(stream.fileno())
            except FileExistsError:
                pass
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".info-discarded-", delete=False) as stream:
                    temporary = Path(stream.name)
                    os.fchmod(stream.fileno(), stat.S_IMODE(path.stat().st_mode))
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                if path.read_bytes() != original:
                    raise ValueError("info.json changed during saving; please retry")
                os.replace(temporary, path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        self.discarded = updated
        return updated


def format_time(milliseconds: int) -> str:
    seconds, millis = divmod(max(0, milliseconds), 1000)
    return f"{seconds:02d}:{millis:03d}"


def launch(dataset: Dataset, initial_episode: int) -> int:
    try:
        from PySide6.QtCore import QEvent, Qt, QUrl
        from PySide6.QtGui import QFont
        from PySide6.QtMultimedia import QMediaPlayer
        from PySide6.QtMultimediaWidgets import QVideoWidget
        from PySide6.QtWidgets import (
            QApplication, QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit,
            QMainWindow, QMessageBox, QPushButton, QSizePolicy, QSlider,
            QSpinBox, QVBoxLayout, QWidget,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install PySide6: python -m pip install -r "
            f"{Path(__file__).with_name('requirements-annotator.txt')}"
        ) from exc

    class Window(QMainWindow):
        def __init__(self):
            super().__init__()
            self.episode = initial_episode
            self.ready = False
            self.dragging_timeline = False
            self.setWindowTitle("Episode review · Discarded episodes")
            self.resize(1200, 830)
            self.setMinimumSize(850, 600)
            self.player = QMediaPlayer(self)
            self.player.setAudioOutput(None)

            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)
            layout.setContentsMargins(28, 22, 28, 18)
            layout.setSpacing(14)

            header = QHBoxLayout()
            brand = QVBoxLayout()
            title = QLabel("EPISODE REVIEW")
            title.setObjectName("brand")
            brand.addWidget(title)
            subtitle = QLabel(dataset.root.name)
            subtitle.setObjectName("muted")
            subtitle.setToolTip(str(dataset.root))
            brand.addWidget(subtitle)
            header.addLayout(brand)
            header.addStretch()
            header.addWidget(QLabel("Episode ID"))
            self.episode_input = QSpinBox()
            self.episode_input.setRange(min(dataset.episodes), max(dataset.episodes))
            self.episode_input.setKeyboardTracking(False)
            self.episode_input.setFixedWidth(105)
            self.episode_input.editingFinished.connect(self.open_entered_episode)
            header.addWidget(self.episode_input)
            self.previous = self.button("←", lambda: self.neighbor(-1))
            self.next = self.button("→", lambda: self.neighbor(1))
            header.addWidget(self.previous)
            header.addWidget(self.next)
            layout.addLayout(header)

            self.details = QLabel()
            self.details.setObjectName("muted")
            layout.addWidget(self.details)
            video_frame = QFrame()
            video_frame.setObjectName("videoFrame")
            video_layout = QVBoxLayout(video_frame)
            video_layout.setContentsMargins(1, 1, 1, 1)
            self.video = QVideoWidget()
            self.video.setMinimumHeight(240)
            self.video.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            video_layout.addWidget(self.video)
            self.player.setVideoOutput(self.video)
            layout.addWidget(video_frame, 1)

            controls = QHBoxLayout()
            self.play = self.button("▶  Play", self.toggle_play)
            controls.addWidget(self.play)
            self.time = QLabel("00:000 / 00:000")
            self.time.setObjectName("time")
            controls.addWidget(self.time)
            controls.addStretch()
            controls.addWidget(QLabel("Camera"))
            self.camera = QComboBox()
            for name in dataset.cameras:
                self.camera.addItem(name.removeprefix("observation.images."), name)
            self.camera.currentIndexChanged.connect(self.load_video)
            controls.addWidget(self.camera)
            controls.addWidget(QLabel("Speed"))
            self.speed = QComboBox()
            for rate in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
                self.speed.addItem(f"{rate:g}×", rate)
            self.speed.setCurrentIndex(3)
            self.speed.currentIndexChanged.connect(
                lambda: self.player.setPlaybackRate(self.speed.currentData())
            )
            controls.addWidget(self.speed)
            layout.addLayout(controls)

            self.timeline = QSlider(Qt.Orientation.Horizontal)
            self.timeline.setRange(0, 0)
            self.timeline.sliderPressed.connect(self.start_seek)
            self.timeline.sliderMoved.connect(self.seek)
            self.timeline.sliderReleased.connect(self.end_seek)
            layout.addWidget(self.timeline)

            actions = QHBoxLayout()
            self.mark_button = self.button("", self.toggle_discarded)
            self.mark_button.setObjectName("primary")
            self.mark_button.setMinimumHeight(65)
            self.mark_button.setMinimumWidth(320)
            actions.addWidget(self.mark_button)
            self.count = QLabel()
            actions.addWidget(self.count)
            actions.addStretch()
            layout.addLayout(actions)
            self.status = QLabel()
            self.status.setObjectName("status")
            layout.addWidget(self.status)
            help_label = QLabel(
                "SPACE  mark / unmark discarded     ENTER  next episode     "
                "R / K  play / pause     E  restart video     Alt+← / →  episode"
            )
            help_label.setObjectName("muted")
            layout.addWidget(help_label)

            self.player.positionChanged.connect(self.update_time)
            self.player.durationChanged.connect(self.update_time)
            self.player.playbackStateChanged.connect(self.playback_changed)
            self.player.mediaStatusChanged.connect(self.media_status)
            self.player.errorOccurred.connect(self.media_error)
            QApplication.instance().installEventFilter(self)
            self.load_episode(initial_episode)

        @staticmethod
        def button(label, callback):
            button = QPushButton(label)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(callback)
            return button

        def eventFilter(self, watched, event):
            if (self.isActiveWindow() and QApplication.activeModalWidget() is None
                    and event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease)):
                focus = QApplication.focusWidget()
                editing = isinstance(focus, (QLineEdit, QSpinBox, QComboBox))
                if event.modifiers() == Qt.KeyboardModifier.NoModifier:
                    if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                        if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
                            if focus is self.episode_input or self.episode_input.isAncestorOf(focus):
                                self.episode_input.interpretText()
                                self.open_entered_episode()
                            elif not editing:
                                self.neighbor(1)
                        return not editing or focus is self.episode_input or self.episode_input.isAncestorOf(focus)
                    if not editing and event.key() == Qt.Key.Key_Space:
                        if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
                            self.toggle_discarded()
                        return True
                    if not editing and event.key() in (Qt.Key.Key_R, Qt.Key.Key_K):
                        if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
                            self.toggle_play()
                        return True
                    if not editing and event.key() == Qt.Key.Key_E:
                        if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
                            self.restart_video()
                        return True
                if (not editing and event.modifiers() == Qt.KeyboardModifier.AltModifier
                        and event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right)):
                    if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
                        self.neighbor(-1 if event.key() == Qt.Key.Key_Left else 1)
                    return True
            return super().eventFilter(watched, event)

        def open_entered_episode(self):
            index = self.episode_input.value()
            if index != self.episode:
                self.load_episode(index)
            self.episode_input.clearFocus()
            self.video.setFocus()

        def neighbor(self, direction):
            index = dataset.episodes.index(self.episode) + direction
            if 0 <= index < len(dataset.episodes):
                self.load_episode(dataset.episodes[index])

        def load_episode(self, episode):
            if episode not in dataset.known:
                self.status.setText(f"Episode {episode} is not in meta/episodes.jsonl")
                self.episode_input.setValue(self.episode)
                return
            self.episode = episode
            self.episode_input.setValue(episode)
            position = dataset.episodes.index(episode)
            self.previous.setEnabled(position > 0)
            self.next.setEnabled(position < len(dataset.episodes) - 1)
            description = dataset.descriptions.get(episode) or "No task label"
            self.details.setText(f"{position + 1} / {len(dataset.episodes)}   ·   {description}")
            self.refresh_mark()
            self.status.setText("Ready · changes are saved automatically")
            self.load_video()

        def load_video(self, *args):
            self.player.stop()
            self.player.setSource(QUrl())
            self.ready = False
            self.play.setEnabled(False)
            self.timeline.setRange(0, 0)
            camera = self.camera.currentData()
            path = dataset.videos.get(camera, {}).get(self.episode)
            if path is None or not path.is_file():
                self.status.setText(f"No {camera} video for episode {self.episode}")
                return
            self.player.setSource(QUrl.fromLocalFile(str(path)))
            self.player.pause()

        def refresh_mark(self):
            marked = self.episode in dataset.discarded
            self.mark_button.setText(
                "✓ DISCARD — click or SPACE to undo" if marked else "SPACE · Mark as discarded"
            )
            self.mark_button.setProperty("marked", marked)
            self.mark_button.style().unpolish(self.mark_button)
            self.mark_button.style().polish(self.mark_button)
            self.count.setText(f"Discarded: {len(dataset.discarded)} / {len(dataset.episodes)}")

        def toggle_discarded(self):
            desired = self.episode not in dataset.discarded
            try:
                dataset.set_discarded(self.episode, desired)
            except (OSError, ValueError, TypeError) as exc:
                self.status.setText("Save failed; the mark was not changed")
                QMessageBox.critical(self, "Could not save", str(exc))
                return
            self.refresh_mark()
            verb = "marked as discarded" if desired else "restored"
            self.status.setText(f"Episode {self.episode} {verb} · saved to meta/info.json")

        def media_status(self, status):
            self.ready = status in (
                QMediaPlayer.MediaStatus.LoadedMedia,
                QMediaPlayer.MediaStatus.BufferedMedia,
                QMediaPlayer.MediaStatus.BufferingMedia,
                QMediaPlayer.MediaStatus.EndOfMedia,
            )
            self.play.setEnabled(self.ready)

        def media_error(self, error, message):
            self.ready = False
            self.play.setEnabled(False)
            self.status.setText(f"Video error: {message}")

        def playback_changed(self, state):
            self.play.setText(
                "Ⅱ  Pause" if state == QMediaPlayer.PlaybackState.PlayingState else "▶  Play"
            )

        def toggle_play(self):
            if not self.ready:
                return
            if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                self.player.pause()
            else:
                if self.player.position() >= self.player.duration():
                    self.player.setPosition(0)
                self.player.play()

        def restart_video(self):
            if self.ready:
                self.player.setPosition(0)
                self.player.play()

        def update_time(self, *args):
            position = self.player.position()
            duration = self.player.duration()
            self.time.setText(f"{format_time(position)} / {format_time(duration)}")
            self.timeline.setMaximum(max(0, duration))
            if not self.dragging_timeline:
                self.timeline.setValue(position)

        def start_seek(self):
            self.dragging_timeline = True
            self.player.pause()

        def seek(self, position):
            if self.ready:
                self.player.setPosition(position)
                self.time.setText(f"{format_time(position)} / {format_time(self.player.duration())}")

        def end_seek(self):
            self.seek(self.timeline.value())
            self.dragging_timeline = False

        def closeEvent(self, event):
            QApplication.instance().removeEventFilter(self)
            self.player.stop()
            self.player.setSource(QUrl())
            super().closeEvent(event)

    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    app.setFont(QFont("DejaVu Sans", 10))
    app.setStyleSheet("""
        QWidget { background: #11161e; color: #e5eaf2; }
        QLabel { background: transparent; }
        QLabel#brand { font-size: 22px; font-weight: 700; letter-spacing: 3px; }
        QLabel#muted { color: #8995a7; font-size: 11px; }
        QLabel#time { font-family: monospace; font-size: 18px; padding-left: 12px; }
        QLabel#status { color: #ffba7c; font-size: 12px; }
        QFrame#videoFrame { background: #080b10; border: 1px solid #2c3542; border-radius: 10px; }
        QPushButton, QSpinBox, QComboBox {
            background: #1c2430; border: 1px solid #323d4e; border-radius: 7px;
            padding: 10px 12px; font-size: 12px;
        }
        QPushButton:hover { background: #293444; border-color: #6c7d93; }
        QPushButton:focus { border-color: #ffac62; }
        QPushButton:disabled { color: #586476; border-color: #26303d; }
        QPushButton#primary { background: #ffac62; color: #1c1b1b; font-weight: 600; }
        QPushButton#primary:hover { background: #ffc18b; }
        QPushButton#primary[marked="true"] { background: #803d3d; color: white; }
        QPushButton#primary[marked="true"]:hover { background: #9b4c4c; }
        QSlider::groove:horizontal { height: 6px; background: #343c48; border-radius: 3px; }
        QSlider::handle:horizontal { width: 16px; margin: -5px 0; background: #ffac62; border-radius: 8px; }
        QMessageBox QPushButton { min-width: 80px; }
    """)
    window = Window()
    window.show()
    previous_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda signum, frame: window.close())
    try:
        return app.exec()
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset", nargs="?", type=Path, default=DEFAULT_DATASET,
        help=f"Dataset root (default: {DEFAULT_DATASET})",
    )
    parser.add_argument("--episode", type=int, help="Episode ID to open at startup")
    args = parser.parse_args()
    try:
        dataset = Dataset(args.dataset)
        episode = dataset.episodes[0] if args.episode is None else args.episode
        if episode not in dataset.known:
            raise ValueError(f"Episode {episode} is not in meta/episodes.jsonl")
        return launch(dataset, episode)
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
