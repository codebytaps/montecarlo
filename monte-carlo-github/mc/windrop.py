from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from pathlib import Path
from typing import Callable

WM_DROPFILES = 0x0233
GWL_WNDPROC = -4
DRAGQUERY_COUNT = 0xFFFFFFFF

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, ctypes.c_uint,
                             wintypes.WPARAM, wintypes.LPARAM)

_INSTALLED: dict[int, tuple] = {}


def _window_handle(root) -> int | None:
    try:
        frame = root.wm_frame()
        if frame:
            return int(frame, 16)
    except Exception:
        pass
    try:
        return int(ctypes.windll.user32.GetParent(root.winfo_id()))
    except Exception:
        return None


def _paths_from_drop(hdrop: int) -> list[Path]:
    shell32 = ctypes.windll.shell32
    count = shell32.DragQueryFileW(wintypes.HANDLE(hdrop), DRAGQUERY_COUNT, None, 0)
    out: list[Path] = []
    for i in range(count):
        length = shell32.DragQueryFileW(wintypes.HANDLE(hdrop), i, None, 0)
        buf = ctypes.create_unicode_buffer(length + 1)
        shell32.DragQueryFileW(wintypes.HANDLE(hdrop), i, buf, length + 1)
        if buf.value:
            out.append(Path(buf.value))
    shell32.DragFinish(wintypes.HANDLE(hdrop))
    return out


def enable_file_drop(root, on_files: Callable[[list[Path]], None]) -> bool:
    if not sys.platform.startswith("win"):
        return False

    root.update_idletasks()
    hwnd = _window_handle(root)
    if not hwnd:
        return False
    if hwnd in _INSTALLED:
        return True

    user32 = ctypes.windll.user32
    shell32 = ctypes.windll.shell32

    set_long = getattr(user32, "SetWindowLongPtrW", None) or user32.SetWindowLongW
    set_long.argtypes = [wintypes.HWND, ctypes.c_int, WNDPROC]
    set_long.restype = ctypes.c_void_p
    user32.CallWindowProcW.argtypes = [ctypes.c_void_p, wintypes.HWND, ctypes.c_uint,
                                       wintypes.WPARAM, wintypes.LPARAM]
    user32.CallWindowProcW.restype = LRESULT

    def handler(hwnd_, msg, wparam, lparam):
        if msg == WM_DROPFILES:
            try:
                paths = _paths_from_drop(wparam)
            except Exception:
                paths = []
            if paths:
                root.after_idle(lambda p=paths: on_files(p))
            return 0
        return user32.CallWindowProcW(old_proc, hwnd_, msg, wparam, lparam)

    callback = WNDPROC(handler)
    old_proc = set_long(wintypes.HWND(hwnd), GWL_WNDPROC, callback)
    if not old_proc:
        return False

    shell32.DragAcceptFiles(wintypes.HWND(hwnd), True)
    _INSTALLED[hwnd] = (callback, old_proc)
    return True
