import os
import sys
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pubsub import pub

from app.src.core import CoreApplication

class PDFHandler(FileSystemEventHandler):
    def __init__(self, app):
        self.app = app
        self._setup_platform_specifics()

    def _setup_platform_specifics(self):
        """Initialize platform-specific components"""
        self.windows_lock_check = False
        self.unix_lock_check = False
        
        if sys.platform == 'win32':
            try:
                import win32con
                import win32file
                import pywintypes
                self.win32con = win32con
                self.win32file = win32file
                self.pywintypes = pywintypes
                self.windows_lock_check = True
                pub.sendMessage('log_event', message="Using Windows file lock detection")
            except ImportError:
                pub.sendMessage('log_event', message="pywin32 not installed, using fallback file checks")
        
        elif sys.platform in ('linux', 'darwin'):
            try:
                import fcntl
                self.fcntl = fcntl
                self.unix_lock_check = True
                pub.sendMessage('log_event', message="Using UNIX file lock detection")
            except ImportError:
                pub.sendMessage('log_event', message="fcntl not available, using fallback file checks")

    def _is_file_locked(self, file_path):
        """Cross-platform file lock detection"""
        if self.windows_lock_check:
            try:
                handle = self.win32file.CreateFile(
                    file_path,
                    self.win32con.GENERIC_READ,
                    0,
                    None,
                    self.win32con.OPEN_EXISTING,
                    self.win32con.FILE_ATTRIBUTE_NORMAL,
                    None
                )
                self.win32file.CloseHandle(handle)
                return False
            except self.pywintypes.error as e:
                if e.winerror == 32:  # ERROR_SHARING_VIOLATION
                    return True
                return False
            except Exception:
                return True
        
        elif self.unix_lock_check:
            try:
                with open(file_path, 'rb') as f:
                    self.fcntl.flock(f, self.fcntl.LOCK_EX | self.fcntl.LOCK_NB)
                    self.fcntl.flock(f, self.fcntl.LOCK_UN)
                return False
            except (IOError, BlockingIOError):
                return True
        
        # Fallback method for unsupported platforms or missing dependencies
        try:
            with open(file_path, 'rb') as f:
                return False
        except IOError as e:
            # Common error codes for locked files across platforms
            if e.errno in (13, 11, 35):  # Permission denied, resource unavailable, etc.
                return True
            return False

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith('.pdf'):
            pub.sendMessage('log_event', message=f"New PDF detected: {event.src_path}")
            if self._wait_for_file(event.src_path):
                self.app.run(event.src_path)

    def _wait_for_file(self, file_path, wait_time=2, max_attempts=30):
        """Wait until file is stable and unlocked"""
        stable_checks = 0
        required_stable_checks = 3
        attempts = 0

        while attempts < max_attempts:
            try:
                if not os.path.exists(file_path):
                    pub.sendMessage('log_event', message=f"File disappeared: {file_path}")
                    return False

                # Size stability check
                initial_size = os.path.getsize(file_path)
                time.sleep(wait_time)
                new_size = os.path.getsize(file_path)

                if initial_size != new_size:
                    stable_checks = 0
                    pub.sendMessage('log_event', message=f"Size changed: {file_path} ({initial_size} → {new_size})")
                else:
                    stable_checks += 1
                    pub.sendMessage('log_event', message=f"Stable check {stable_checks}/{required_stable_checks}")

                # Lock check and final verification
                if stable_checks >= required_stable_checks:
                    if not self._is_file_locked(file_path):
                        # Final read test
                        try:
                            with open(file_path, 'rb') as f:
                                f.read(1)
                            pub.sendMessage('log_event', message=f"File ready: {file_path}")
                            return True
                        except IOError as e:
                            pub.sendMessage('log_event', message=f"Final check failed: {str(e)}")
                    else:
                        pub.sendMessage('log_event', message=f"File still locked: {file_path}")

                attempts += 1

            except Exception as e:
                pub.sendMessage('log_event', message=f"Error checking {file_path}: {str(e)}")
                attempts += 1
                time.sleep(wait_time)

        pub.sendMessage('log_event', message=f"Failed to process: {file_path}")
        return False

class Watcher:
    def __init__(self, openai_api_key, language, csv_dir):
        self.openai_api_key = openai_api_key
        self.language = language
        self.csv_dir = csv_dir
        self.observer = Observer()

    def start(self, output_dir: str, folder_to_watch: str, stop_event):
        pub.sendMessage('log_event', message=f"Watching folder: {folder_to_watch}")
        self.app = CoreApplication(self.openai_api_key, self.language, self.csv_dir, output_dir)
        self.event_handler = PDFHandler(self.app)
        self.observer.schedule(self.event_handler, folder_to_watch, recursive=False)
        self.observer.start()
        try:
            while not stop_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()
        finally:
            self.observer.stop()
            self.observer.join()