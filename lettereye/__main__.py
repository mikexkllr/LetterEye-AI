import multiprocessing

from lettereye.main import main

if __name__ in {"__main__", "__mp_main__"}:
    multiprocessing.freeze_support()
    if __name__ == "__main__":
        main()
