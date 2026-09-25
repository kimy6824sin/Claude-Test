"""Application entry point: ``python -m meshrev [files...] [--demo]``."""

from __future__ import annotations

import argparse
import logging
import os
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meshrev", description="活塞零件网格逆向建模工具")
    parser.add_argument("files", nargs="*", help="启动时导入的 STL/OBJ/PLY/STEP 文件")
    parser.add_argument("--demo", action="store_true", help="启动时生成并打开示例活塞")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    args, qt_args = parser.parse_known_args(argv if argv is not None else sys.argv[1:])

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    os.environ.setdefault("QT_API", "pyside6")

    from PySide6.QtWidgets import QApplication

    from meshrev.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("MeshRev")
    app.setOrganizationName("meshrev")
    window = MainWindow()
    window.show()
    if args.files:
        window.import_files(args.files)
    if args.demo:
        window.controller.load_demo_piston()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
