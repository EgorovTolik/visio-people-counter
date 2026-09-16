"""Qt-бэкенд окна калибратора (PySide6) — задача 16.

Окно поверх ТОГО ЖЕ :class:`~visio_people_counter.calib_controller.CalibrationController`,
что и cv2-драйвер (:func:`calibrate.run_calibration`): логика (режимы, кнопки, seek,
ROI, save) не дублируется — Qt-окно только рисует кадр и пересылает события.

* :class:`VideoCanvas` — canvas кадра (BGR ndarray → QPixmap; размер =
  ``width*scale × height*scale``); клики мыши переводятся через
  :func:`calibrate.unscale_mouse` в координаты исходного кадра;
* :class:`CalibrateQtWindow` — QMainWindow: QToolBar с 13 кнопками над кадром
  (подписи и имена ``on_button`` — ИМЕННО как у cv2-панели), статусбар
  («счётчик: … | кадр N | scale=…x | <последнее сообщение>»), QTimer-цикл с
  интервалом ``1000/fps_источника`` (fps<=0 → 33 мс); тик вынесен в :meth:`tick`
  (вызывается таймером И вручную в тестах);
* :class:`CountQtWindow` — окно подсчёта поверх :class:`~visio_people_counter.gui.GuiPlayer`
  (задача 17): QToolBar «Пауза/+скорость/−скорость/масштаб ↓/масштаб ↑/Выход»
  вызывает ``player.handle_key(...)`` с кодами cv2-клавиш; статусбар —
  ``player.status_text(...)`` + последнее сообщение; QTimer с интервалом
  ``1000/(fps*speed)`` (fps<=0 → 33 мс), тик — :meth:`CountQtWindow.tick_once`;
* :func:`run_calibration_qt` — зеркальный по смыслу ``run_calibration``;
* :func:`run_count_qt` — Qt-драйвер ``count --gui --backend qt``.

Импорт PySide6 — ТОЛЬКО в этом модуле (лениво из CLI/тестов). ``apply_qt_env()``
из gui.py для PySide6 НЕ вызывается: он настраивает QT_PLUGIN_PATH под bundled-Qt
колеса opencv, что конфликтует с собственными плагинами PySide6.

Запуск: ``python -m visio_people_counter calibrate --backend qt`` (qt — по
умолчанию при установленном PySide6; без него CLI фолбэкает на cv2).
"""

from __future__ import annotations

import sys
from typing import Callable, Optional

import numpy as np

from PySide6.QtCore import QPoint, Qt, QTimer
# QAction — в QtGui с Qt6 (в QtWidgets только deprecated-алиас)
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow, QToolBar, QWidget

import cv2   # BGR→RGB для canvas; обязательная зависимость проекта (не Qt-специфична)


def _sanitize_env_for_pyside() -> None:
    """Убрать из окружения настройки, сделанные под bundled-Qt колеса OpenCV.

    ``apply_qt_env`` (gui.py) добавляет в QT_PLUGIN_PATH системные плагины Qt5 —
    для PySide6 (Qt6) они несовместимы: xcb-плагин Qt5 не грузится в Qt6 и окно
    падает. PySide6 использует СВОИ плагины из wheel — чужие пути не нужны.
    Вызывается ДО создания QApplication (в обоих entry points).
    """
    import os as _os
    cur = _os.environ.get("QT_PLUGIN_PATH", "")
    keep = [p for p in cur.split(_os.pathsep)
            if p and ("/cv2/qt" not in p) and ("qt5/plugins" not in p)]
    if keep:
        _os.environ["QT_PLUGIN_PATH"] = _os.pathsep.join(keep)
    else:
        _os.environ.pop("QT_PLUGIN_PATH", None)
    # тема берётся стандартным механизмом Qt6; принудительный gtk3 из cv2-пути не нужен
    _os.environ.pop("QT_QPA_PLATFORMTHEME", None)

from .pipeline import Pipeline   # для аннотации run_count_qt

from .calibrate import (DEFAULT_CACHE_FRAMES, load_calibration_config,
                        resolve_save_target, unscale_mouse)
from .gui import GuiPlayer


# ---------------------------------------------------------------------------
# Клавиатура: QKeyEvent → код cv2.waitKey(1)&0xFF для controller.on_key
# ---------------------------------------------------------------------------

#: специальные клавиши → код в формате ``waitKey & 0xFF`` (как в cv2-драйвере).
_SPECIAL_KEYS = {
    int(Qt.Key_Return): 13,     # Enter (основной блок)
    int(Qt.Key_Enter): 13,      # Enter (цифровая клавиатура)
    int(Qt.Key_Escape): 27,
    int(Qt.Key_Backspace): 8,
    int(Qt.Key_Space): 32,
}


def qt_key_code(key: int, text: str,
                modifiers=Qt.KeyboardModifier.NoModifier) -> Optional[int]:
    """Вынесенный маппинг ``(QKeyEvent.key(), QKeyEvent.text())`` → код ``on_key``.

    Чистая функция (без объекта события — удобно тестировать):

    * печатаемые символы → ``ord(text)`` (строчные l/z/s/m/r/n/p/a/x/b/v/t/q,
      цифры 0-9, `,`/`.` и т.п.) — то же, что ``cv2.waitKey(1) & 0xFF``;
    * Enter → 13, ESC → 27, Backspace → 8, Space → 32 (таблица :data:`_SPECIAL_KEYS`);
    * Ctrl/Alt-комбинации и нераспознанные клавиши → ``None`` (игнорируются).

    :param key: значение ``QKeyEvent.key()`` (int).
    :param text: ``QKeyEvent.text()`` (строка печатаемого символа; может быть "").
    :param modifiers: ``QKeyEvent.modifiers()``.
    """
    if modifiers & (Qt.KeyboardModifier.ControlModifier
                    | Qt.KeyboardModifier.AltModifier):
        return None
    mapped = _SPECIAL_KEYS.get(int(key))
    if mapped is not None:
        return mapped
    if text and len(text) == 1:
        o = ord(text[0])
        # только печатаемый ASCII — как waitKey&0xFF (кириллица/контрольные — мимо)
        if 32 < o < 127:
            return o
    return None


def qt_key_to_cv2(key_event) -> Optional[int]:
    """QKeyEvent → код, ожидаемый ``controller.on_key`` (формат waitKey&0xFF).

    :param key_event: объект QKeyEvent.
    :returns: ASCII-код клавиши или ``None`` (клавиша игнорируется).
    """
    return qt_key_code(int(key_event.key()), key_event.text(), key_event.modifiers())


# ---------------------------------------------------------------------------
# Canvas кадра
# ---------------------------------------------------------------------------

class VideoCanvas(QWidget):
    """Canvas кадра калибратора.

    * :meth:`show_frame` — BGR ndarray → QPixmap (конвертация в RGB; bytesPerLine
      = ``w*3``); размер виджета = ``(int(width*scale), int(height*scale))`` —
      пересчитывается при каждом кадре (width/height/scale контроллера меняются
      при `,`/`.` и принятии ROI);
    * :meth:`mousePressEvent` / :meth:`mouseMoveEvent` — координаты события →
      :func:`calibrate.unscale_mouse(x, y, scale, w, h)` → колбэки ``on_click`` /
      ``on_mouse_move`` (устанавливает окно: ``ctrl.on_click``/``ctrl.on_mouse_move``).
    """

    def __init__(self, ctrl, parent=None):
        super().__init__(parent)
        self._ctrl = ctrl
        #: колбэки в СИСТЕМЕ ИСХОДНОГО кадра (устанавливает CalibrateQtWindow)
        self.on_click: Optional[Callable[[int, int], None]] = None
        self.on_mouse_move: Optional[Callable[[int, int], None]] = None
        self._frame: Optional[np.ndarray] = None   # последний BGR-кадр (удерживаем)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)   # клавиатуру обрабатывает окно
        self._update_size()

    def _update_size(self) -> None:
        """Размер виджета = исходный кадр × scale контроллера."""
        w = max(1, int(self._ctrl.width * self._ctrl.scale))
        h = max(1, int(self._ctrl.height * self._ctrl.scale))
        self.setFixedSize(w, h)

    def show_frame(self, img: np.ndarray) -> None:
        """Показать BGR-кадр (исходного разрешения; resize под scale НЕ нужен —
        размер canvas уже учитывает scale)."""
        if img is None or img.size == 0:
            return
        self._frame = img
        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        # QPixmap копирует пиксели — кадр в кэше можно менять/освобождать
        self._pixmap = QPixmap.fromImage(qimg)
        self._update_size()   # width/height/scale могли измениться (ROI / `,` `.`)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (имя Qt)
        """Отрисовать последний кадр (до первого кадра — чёрный фон)."""
        from PySide6.QtGui import QPainter
        p = QPainter(self)
        pm = getattr(self, "_pixmap", None)
        if pm is not None and not pm.isNull():
            # pixmap в исходном разрешении; растягиваем на размер виджета (×scale)
            p.drawPixmap(0, 0, int(self.width()), int(self.height()), pm)
        else:
            p.fillRect(self.rect(), Qt.GlobalColor.black)
        p.end()

    def _to_source_coords(self, pos: QPoint) -> tuple[int, int]:
        """Координаты события на canvas → координаты исходного кадра."""
        return unscale_mouse(pos.x(), pos.y(), self._ctrl.scale,
                             self._ctrl.width, self._ctrl.height)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.on_click is not None:
            x, y = self._to_source_coords(event.position().toPoint())
            self.on_click(x, y)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.on_mouse_move is not None:
            x, y = self._to_source_coords(event.position().toPoint())
            self.on_mouse_move(x, y)
        super().mouseMoveEvent(event)


# ---------------------------------------------------------------------------
# Окно калибратора
# ---------------------------------------------------------------------------

#: Кнопки тулбара: (name для ctrl.on_button, подпись, checkable).
#: ИМЕННО те же имена и подписи, что у cv2-панели (calibrate._BUTTON_LABELS):
#: активный режим/маска/«все» — подсветка через QAction.setChecked.
_BUTTON_DEFS: list[tuple[str, str, bool]] = [
    ("line", "линия", True),
    ("zone", "зона", True),
    ("size", "размер", True),
    ("roi", "roi", True),
    ("mask", "маска", True),
    ("show_all", "все", True),
    ("prev", "<", False),
    ("next", ">", False),
    ("new_line", "+линия", False),
    ("new_zone", "+зона", False),
    ("delete", "удалить", False),
    ("time", "время", False),
    ("save", "сохранить", False),
]


class CalibrateQtWindow(QMainWindow):
    """QMainWindow калибратора: canvas + QToolBar (13 кнопок) + статусбар.

    Цикл — QTimer с интервалом ``1000/fps_источника`` (fps<=0 → 33 мс), как
    cv2-реалтайм; один тик = :meth:`tick` (ctrl.step() → canvas + статусбар;
    None/quit_requested → closeWindow()). Метод ``tick()`` вызывается и таймером,
    и вручную (тесты). Клавиатура — единая точка: keyPressEvent ОКНА.
    """

    def __init__(self, ctrl):
        super().__init__()
        self.ctrl = ctrl
        self._closed = False   # QWidget.isClosed() нет — состояние ведём сами (closeEvent)
        self.setWindowTitle(ctrl.window_name)

        # центральный виджет — canvas; мышь → координаты исходного кадра → ctrl
        self.canvas = VideoCanvas(ctrl, self)
        self.canvas.on_click = ctrl.on_click
        self.canvas.on_mouse_move = ctrl.on_mouse_move
        self.setCentralWidget(self.canvas)

        # верхняя панель: QToolBar с кнопками (подписи/имена — как у cv2-панели)
        toolbar = QToolBar("Калибровка", self)
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        self.addToolBar(toolbar)
        self.actions: dict[str, QAction] = {}
        for name, label, checkable in _BUTTON_DEFS:
            act = QAction(label, self)
            act.setCheckable(checkable)
            act.triggered.connect(
                lambda _checked=False, n=name: self._on_action(n))
            toolbar.addAction(act)
            self.actions[name] = act

        # статусбар: «счётчик: <id> | кадр N | scale=…x | <последнее сообщение>»
        self.statusBar().showMessage(self._status_text())

        # цикл по таймеру: реалтайм fps источника (как cv2-драйвер); fps<=0 → 33 мс
        fps = 0.0
        pipe = getattr(ctrl, "_pipe", None)
        source = getattr(pipe, "source", None) if pipe is not None else None
        if source is not None:
            fps = float(getattr(source, "fps", 0.0) or 0.0)
        interval_ms = int(round(1000.0 / fps)) if fps > 0 else 33
        self.timer = QTimer(self)
        self.timer.setInterval(max(1, min(interval_ms, 1000)))
        self.timer.timeout.connect(self.tick)

    # ------------------------------------------------------------------ действия
    def _on_action(self, name: str) -> None:
        """Нажатие кнопки тулбара — прямое действие по имени (как клик в cv2)."""
        self.ctrl.on_button(name)
        self._refresh_ui()

    def tick(self) -> None:
        """Один цикл: ``img = ctrl.step()`` → canvas + статусбар.

        ``None``/``quit_requested`` → :meth:`closeWindow`. Вызывается QTimer'ом
        и вручную (тесты). Кадр не «листается» сам — листание [n]/[p] через
        клавиатуру, как в cv2-окне.
        """
        img = self.ctrl.step()
        if img is None or self.ctrl.quit_requested:
            self.closeWindow()
            return
        self.canvas.show_frame(img)
        self._refresh_ui()

    def _refresh_ui(self) -> None:
        """Синхронизировать подсветку кнопок и статусбар с состоянием контроллера."""
        c = self.ctrl
        active = {
            "line": c.state.mode == "line",
            "zone": c.state.mode == "zone",
            "size": c.state.mode == "size",
            "roi": c.state.mode == "roi",
            "mask": bool(c.mask_on),
            "show_all": bool(c.show_all),
        }
        for name, act in self.actions.items():
            if act.isCheckable() and act.isChecked() != active[name]:
                act.blockSignals(True)   # не дёргать on_button при пересинхронизации
                act.setChecked(active[name])
                act.blockSignals(False)
        self.statusBar().showMessage(self._status_text())

    def _status_text(self) -> str:
        c = self.ctrl
        txt = (f"счётчик: {c.state.counter_id} | кадр {c.frame_index} "
               f"| scale={c.scale:g}x")
        msgs = c.messages   # активные уведомления БЕЗ отрисовки (TTL 5 c)
        if msgs:
            txt += f" | {msgs[-1]}"
        return txt

    def closeWindow(self) -> None:
        """Закрыть окно (closeEvent закроет источник контроллера)."""
        self.close()

    # ------------------------------------------------------------------ события
    def keyPressEvent(self, event):
        """Клавиатура — единая точка окна: QKeyEvent → код on_key."""
        code = qt_key_to_cv2(event)
        if code is not None:
            self.ctrl.on_key(code)
            self._refresh_ui()
        else:
            super().keyPressEvent(event)

    @property
    def is_closed(self) -> bool:
        """Окно закрыто (closeEvent сработал)."""
        return self._closed

    def closeEvent(self, event):
        """Выход: остановить таймер, закрыть источник, финальный print."""
        self._closed = True
        self.timer.stop()
        try:
            # источник закрывается строго ПОСЛЕ закрытия окна (seek [t] работает)
            self.ctrl.close()
        finally:
            if not self.ctrl.saved_once:
                print("calibrate: выход БЕЗ сохранения ([a] не нажимался "
                      "или изменений не было)")
        event.accept()


# ---------------------------------------------------------------------------
# Окно подсчёта (count --gui --backend qt) — задача 17
# ---------------------------------------------------------------------------

class _CountCanvasState:
    """Состояние отображения для :class:`VideoCanvas` в окне счёта.

    Canvas здесь только показывает кадр (координаты мыши НЕ используются), поэтому
    вместо контроллера передаётся лёгкий объект с ``width``/``height``/``scale``:
    размеры последнего кадра, уже отмасштабированного ``GuiPlayer.tick()``
    (``scale = 1.0`` — размер виджета совпадает с pixmap-ом; эффективный масштаб
    окна = ``player.scale`` и меняется вместе с пресетами `,`/`.`).
    """

    def __init__(self) -> None:
        self.width: int = 1
        self.height: int = 1
        self.scale: float = 1.0


#: Кнопки тулбара окна счёта: (name, подпись, код клавиши cv2, checkable).
#: Коды — те же, что у клавиш cv2-окна: пробел/+/−/,/. /q.
_COUNT_BUTTON_DEFS: list[tuple[str, str, int, bool]] = [
    ("pause", "Пауза", 32, True),
    ("faster", "+скорость", ord("+"), False),
    ("slower", "−скорость", ord("-"), False),
    ("scale_down", "масштаб ↓", ord(","), False),
    ("scale_up", "масштаб ↑", ord("."), False),
    ("quit", "Выход", ord("q"), False),
]


class CountQtWindow(QMainWindow):
    """QMainWindow окна подсчёта: canvas + QToolBar (6 кнопок) + статусбар.

    Логика НЕ дублируется — всё состояние живёт в :class:`GuiPlayer` (общий с
    cv2-режимом): кнопки вызывают ``player.handle_key(code)`` с кодами cv2
    (пробел/+/−/,/. /q). Цикл — QTimer с интервалом ``1000/(fps*speed)``
    (пересчитывается при смене speed; fps<=0 → 33 мс); пауза таймер НЕ
    останавливает — ``player.tick()`` сам возвращает последний кадр (как cv2).
    Тик — :meth:`tick_once` (вызывается таймером И вручную в тестах):
    ``None`` из ``tick()`` → ``player.finalize(...)`` + :meth:`closeWindow`.
    """

    def __init__(self, player: GuiPlayer):
        super().__init__()
        self.player = player
        self._closed = False   # QWidget.isClosed() нет — состояние ведём сами (closeEvent)
        self.setWindowTitle(player.window_name)

        # центральный виджет — тот же VideoCanvas; только отображение, мышь не нужна
        self.canvas_state = _CountCanvasState()
        self.canvas = VideoCanvas(self.canvas_state, self)
        self.setCentralWidget(self.canvas)

        # верхняя панель: QToolBar с кнопками (коды — как клавиши cv2-окна)
        toolbar = QToolBar("Подсчёт", self)
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        self.addToolBar(toolbar)
        self.actions: dict[str, QAction] = {}
        for name, label, key, checkable in _COUNT_BUTTON_DEFS:
            act = QAction(label, self)
            act.setCheckable(checkable)
            act.triggered.connect(
                lambda _checked=False, n=name: self._on_button(n))
            toolbar.addAction(act)
            self.actions[name] = act

        # статусбар: строка статуса плеера + последнее сообщение; обновляется каждый тик
        self.statusBar().showMessage(self._status_text())

        # цикл по таймеру: реалтайм fps/speed (как cv2-драйвер);
        # fps неизвестен до первого tick → 33 мс, далее пересчёт в _refresh_ui
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.tick_once)

    # ------------------------------------------------------------------ действия
    @staticmethod
    def _key_of(name: str) -> int:
        for n, _label, key, _checkable in _COUNT_BUTTON_DEFS:
            if n == name:
                return key
        raise KeyError(name)

    def _on_button(self, name: str) -> None:
        """Нажатие кнопки тулбара → ``player.handle_key(код cv2-клавиши)``."""
        self.player.handle_key(self._key_of(name))
        self._refresh_ui()

    def tick_once(self) -> None:
        """Один тик: ``view = player.tick()`` → canvas + статусбар.

        ``None`` (EOF/stop) → ``player.finalize(...)`` + :meth:`closeWindow`.
        Вызывается QTimer'ом и вручную (тесты).
        """
        view = self.player.tick()
        if view is None:
            self.player.finalize("окно закрыто (GUI)")
            self.closeWindow()
            return
        st = self.canvas_state
        h, w = int(view.shape[0]), int(view.shape[1])
        st.width, st.height, st.scale = w, h, 1.0   # кадр уже отмасштабирован player'ом
        self.canvas.show_frame(view)
        self._refresh_ui()

    def _refresh_ui(self) -> None:
        """Синхронизировать подсветку «Пауза», интервал таймера и статусбар."""
        p = self.player
        act = self.actions["pause"]
        if act.isChecked() != p._paused:
            act.blockSignals(True)   # не дёргать handle_key при пересинхронизации
            act.setChecked(p._paused)
            act.blockSignals(False)
        src = getattr(p.pipeline, "source", None)
        fps = float(getattr(src, "fps", 0.0) or 0.0) if src is not None else 0.0
        interval = (int(round(1000.0 / (fps * p.speed)))
                    if fps > 0 and p.speed > 0 else 33)
        self.timer.setInterval(max(1, min(interval, 1000)))
        self.statusBar().showMessage(self._status_text())

    def _status_text(self) -> str:
        """Строка статусбара: ``GuiPlayer.status_text`` + последнее сообщение."""
        p = self.player
        proc_fps = (1.0 / p._proc_ema) if p._proc_ema > 0 else 0.0
        txt = p.status_text(proc_fps, frame_index=p.frame_index)
        if p.last_message:
            txt += f" | {p.last_message}"
        return txt

    def closeWindow(self) -> None:
        """Закрыть окно (closeEvent остановит таймер и завершит плеер)."""
        self.close()

    # ------------------------------------------------------------------ события
    def keyPressEvent(self, event):
        """Клавиатура — та же, что у cv2-окна: QKeyEvent → handle_key(код)."""
        code = qt_key_to_cv2(event)
        if code is not None:
            self.player.handle_key(code)
            self._refresh_ui()
        else:
            super().keyPressEvent(event)

    @property
    def is_closed(self) -> bool:
        """Окно закрыто (closeEvent сработал)."""
        return self._closed

    def closeEvent(self, event):
        """Выход: остановить таймер; финализация — идемпотентна (tick_once может
        вызвать её раньше при EOF/«Выход»)."""
        self._closed = True
        self.timer.stop()
        try:
            self.player.finalize("окно закрыто (GUI)")
        finally:
            event.accept()


# ---------------------------------------------------------------------------
# Драйвер count --gui (задача 17)
# ---------------------------------------------------------------------------

def run_count_qt(pipeline: Pipeline, speed: float = 1.0,
                 initial_scale: float = 1.0) -> int:
    """Подсчёт в Qt-окне (PySide6): ``count --gui --backend qt``.

    Создаёт QApplication (если нет), :class:`GuiPlayer` (без запуска cv2-run!) и
    :class:`CountQtWindow`; цикл — QTimer окна, завершение — EOF/«Выход»
    (``player.finalize`` в closeEvent). ``apply_qt_env()`` НЕ вызывается: он
    настраивает QT_PLUGIN_PATH под bundled-Qt колеса opencv, что конфликтует с
    плагинами PySide6.

    :raises ImportError: PySide6 не импортируется (CLI проверяет заранее).
    :returns: 0 — корректное завершение.
    """
    _sanitize_env_for_pyside()
    app = QApplication.instance() or QApplication(sys.argv)
    player = GuiPlayer(pipeline, speed=speed, initial_scale=initial_scale)
    win = CountQtWindow(player)
    win.show()
    win.timer.start()
    try:
        app.exec()
    finally:
        # closeEvent уже остановил таймер и завершил плеер; страховка — идемпотентно
        if not win.is_closed:
            win.close()
    return 0


# ---------------------------------------------------------------------------
# Драйвер (зеркальный по смыслу calibrate.run_calibration)
# ---------------------------------------------------------------------------

def run_calibration_qt(config_path: str | Path, video: Optional[str] = None,
                       counter_id: str = "main_line",
                       save_to: Optional[str | Path] = None,
                       cache_frames: int = DEFAULT_CACHE_FRAMES,
                       initial_scale: float = 1.0) -> int:
    """Интерактивная калибровка в Qt-окне (PySide6).

    Зеркальная по смыслу сигнатура :func:`calibrate.run_calibration`: тот же
    контроллер (загрузка конфига/цель сохранения — общие helpers), `ctrl.open()`,
    QApplication (создаётся, если нет), окно; ``app.exec()`` → 0.

    :raises ImportError: PySide6 не импортируется (CLI проверяет заранее через
        ``importlib.util.find_spec("PySide6")``).
    :returns: 0 — корректное завершение (запись опциональна); 1 — ошибка конфига/источника.
    """
    if cache_frames < 1:
        raise ValueError(f"cache_frames: ожидалось >= 1, получено {cache_frames!r}")

    cfg, _err = load_calibration_config(config_path)
    if cfg is None:
        return 1
    if video:
        cfg.video.path = video
    if not cfg.video.path:
        print("calibrate: не задан источник видео — укажите --video или video.path в конфиге",
              file=sys.stderr)
        return 1
    save_target = resolve_save_target(cfg, config_path, save_to)

    from .calib_controller import CalibrationController   # лениво: как в cv2-драйвере
    from .config import ConfigError
    from .video_source import VideoSourceError

    try:
        ctrl = CalibrationController(
            cfg, counter_id=counter_id, cache_frames=cache_frames,
            initial_scale=initial_scale,
            window_name="calibrate (Qt) [l/z/s/m/a/q]", save_to=save_target)
        ctrl.open()   # Pipeline + кэш первых кадров + seek к processing.frame_start
    except (VideoSourceError, ConfigError):
        # ошибка источника / ни одного кадра: сообщение уже в stderr; источник — до возврата
        ctrl.close()
        return 1

    _sanitize_env_for_pyside()
    app = QApplication.instance() or QApplication(sys.argv)
    win = CalibrateQtWindow(ctrl)
    win.show()
    win.timer.start()
    try:
        app.exec()
    finally:
        # closeEvent уже остановил таймер и закрыл источник; страховка — идемпотентно
        if not win.is_closed:
            win.close()
        ctrl.close()
    return 0
