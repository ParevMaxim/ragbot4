from __future__ import annotations

import os
import sys
import shutil
import subprocess
import threading
import time
from pathlib import Path
from urllib.request import urlretrieve, urlopen

import tkinter as tk
from tkinter import ttk, messagebox

# Можно использовать ваш bootstrap_ollama.py (он у вас уже есть в src)
from bootstrap_ollama import ensure_ollama_running, find_ollama_exe


APP_NAME = "RAG Bot"
APP_EXE_NAME = "RAGChatBot.exe"
INSTALL_DIR_NAME = "ragbot"

OLLAMA_DOWNLOAD_URL = "https://ollama.com/download/OllamaSetup.exe"

MODELS_TO_PULL = ["llama3.1", "nomic-embed-text"]
OLLAMA_HOST = "http://127.0.0.1:11434"


def resource_path(relative_path: str) -> str:
    # для PyInstaller
    try:
        base = sys._MEIPASS  # type: ignore[attr-defined]
    except Exception:
        base = os.path.abspath(".")
    return os.path.join(base, relative_path)


def create_shortcut_vbs(target_exe: str, shortcut_path: str, description: str):
    vbs_script = f"""
Set oWS = WScript.CreateObject("WScript.Shell")
Set oLink = oWS.CreateShortcut("{shortcut_path}")
oLink.TargetPath = "{target_exe}"
oLink.Description = "{description}"
oLink.WorkingDirectory = "{os.path.dirname(target_exe)}"
oLink.Save
"""
    vbs_file = os.path.join(os.getenv("TEMP", "."), "create_shortcut.vbs")
    with open(vbs_file, "w", encoding="utf-8") as f:
        f.write(vbs_script)
    subprocess.call(["cscript", "/nologo", vbs_file])
    try:
        os.remove(vbs_file)
    except Exception:
        pass


def is_ollama_installed() -> bool:
    return bool(find_ollama_exe() or shutil.which("ollama"))


def download_and_install_ollama(log_fn):
    log_fn("Ollama не найдена. Скачиваю установщик...")
    tmp = Path(os.getenv("TEMP", ".")) / "OllamaSetup.exe"
    urlretrieve(OLLAMA_DOWNLOAD_URL, tmp)

    log_fn("Запускаю установку Ollama (может попросить подтверждение)...")
    subprocess.run([str(tmp)], check=True)
    log_fn("Установка Ollama завершена.")


def write_env_file(install_dir: Path):
    env_text = f"""\
OLLAMA_HOST={OLLAMA_HOST}

CHAT_MODEL_MAIN=llama3.1
CHAT_MODEL_SECONDARY=llama3.1
AGGREGATE_MODEL=llama3.1
REWRITE_MODEL=llama3.1

EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIM=768

DOC_LANGUAGE=same
KB_DIR=./data/kb

ENABLE_QUERY_REWRITE=0
ENABLE_FACTOID_BOOST=1
ENABLE_MMR=1

ANSWER_WITH_CITATIONS=1
ENABLE_YEAR_GUARD=1

LLM_NUM_PREDICT=256
LLM_TEMPERATURE=0
LLM_TOP_P=0.9
LLM_NUM_CTX=4096
OLLAMA_KEEP_ALIVE=10m
"""
    (install_dir / ".env").write_text(env_text, encoding="utf-8")


def pull_models_via_cli(log_fn):
    """
    Тянем модели через `ollama pull` (самый простой и надежный способ).
    Предполагается, что `ollama serve` уже запущен.
    """
    ollama = find_ollama_exe() or "ollama"
    for m in MODELS_TO_PULL:
        log_fn(f"Скачивание модели: {m} (это может занять время)...")
        # без shell=True, чтобы было предсказуемо
        subprocess.run([ollama, "pull", m], check=True)
        log_fn(f"Модель {m}: готово.")


class InstallerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"Установка {APP_NAME}")
        self.geometry("520x420")
        self.resizable(False, False)

        self.label = tk.Label(self, text=f"Установка {APP_NAME}", font=("Segoe UI", 16, "bold"))
        self.label.pack(pady=16)

        self.status_label = tk.Label(self, text="Готово к установке", font=("Segoe UI", 10), wraplength=480)
        self.status_label.pack(pady=6)

        self.progress = ttk.Progressbar(self, orient="horizontal", length=460, mode="determinate")
        self.progress.pack(pady=12)

        self.log_text = tk.Text(self, height=14, width=64, font=("Consolas", 8), state="disabled")
        self.log_text.pack(pady=8)

        self.install_btn = tk.Button(
            self,
            text="Начать установку",
            command=self.start_installation,
            bg="#007bff",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            padx=20,
            pady=6
        )
        self.install_btn.pack(pady=10)

    def log(self, message: str):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")
        self.status_label.config(text=message)
        self.update()

    def start_installation(self):
        self.install_btn.config(state="disabled")
        threading.Thread(target=self.run_install_process, daemon=True).start()

    def run_install_process(self):
        try:
            local_app_data = os.getenv("LOCALAPPDATA") or str(Path.home())
            install_dir = Path(local_app_data) / "Programs" / INSTALL_DIR_NAME
            app_exe_dst = install_dir / APP_EXE_NAME

            # 1) create dirs
            self.progress["value"] = 5
            self.log(f"Создаю папку установки: {install_dir}")
            install_dir.mkdir(parents=True, exist_ok=True)

            data_dir = install_dir / "data" / "kb"
            data_dir.mkdir(parents=True, exist_ok=True)
            self.log(f"Создаю папку данных KB: {data_dir}")

            # 2) copy app exe
            self.progress["value"] = 20
            self.log("Копирую файл приложения...")
            src_exe = Path(resource_path(os.path.join("installer_assets", "RAGChatBot.exe")))
            if not src_exe.exists():
                raise FileNotFoundError(f"Не найден installer_assets/RAGChatBot.exe внутри установщика: {src_exe}")
            shutil.copy2(str(src_exe), str(app_exe_dst))

            # 3) write .env
            self.progress["value"] = 30
            self.log("Создаю .env рядом с приложением...")
            write_env_file(install_dir)

            # 4) shortcut
            self.progress["value"] = 40
            self.log("Создаю ярлык на рабочем столе...")
            desktop = Path(os.environ["USERPROFILE"]) / "Desktop"
            shortcut_path = str(desktop / f"{APP_NAME}.lnk")
            create_shortcut_vbs(str(app_exe_dst), shortcut_path, f"Запустить {APP_NAME}")

            # 5) install ollama if missing
            self.progress["value"] = 50
            if not is_ollama_installed():
                download_and_install_ollama(self.log)
            else:
                self.log("Ollama уже установлена.")

            # 6) ensure ollama running
            self.progress["value"] = 60
            self.log("Проверяю запуск Ollama...")
            ok = ensure_ollama_running(OLLAMA_HOST, wait_seconds=30)
            if not ok:
                raise RuntimeError("Не удалось запустить/подключиться к Ollama за 30 секунд.")

            # 7) pull models
            self.progress["value"] = 70
            self.log("Скачиваю модели (llama3.1 + nomic-embed-text)...")
            pull_models_via_cli(self.log)

            self.progress["value"] = 100
            self.log("Установка завершена.")
            messagebox.showinfo("Готово", f"{APP_NAME} установлен.\n\nПапка: {install_dir}")
            self.destroy()

        except Exception as e:
            messagebox.showerror("Ошибка установки", str(e))
            self.install_btn.config(state="normal")


if __name__ == "__main__":
    app = InstallerApp()
    app.mainloop()