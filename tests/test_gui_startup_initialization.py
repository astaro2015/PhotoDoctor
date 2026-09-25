from pathlib import Path
import ast


def _source() -> str:
    return (Path(__file__).parents[1] / "src" / "photodoctor" / "gui" / "main_window.py").read_text(encoding="utf-8")


def _image_view_class(tree: ast.AST) -> ast.ClassDef:
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.ClassDef) and node.name == "ImageView":
            return node
    raise AssertionError("ImageView class not found")


def _method(cls: ast.ClassDef, name: str):
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def test_image_view_event_filter_state_initialized_before_install():
    source = _source()
    tree = ast.parse(source)
    cls = _image_view_class(tree)
    init = _method(cls, "__init__")
    event_filter = _method(cls, "eventFilter")

    install_line = None
    assigned_before_install = set()
    for node in ast.walk(init):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "installEventFilter":
            install_line = node.lineno if install_line is None else min(install_line, node.lineno)
    assert install_line is not None

    for node in ast.walk(init):
        if getattr(node, "lineno", 10**9) >= install_line:
            continue
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr.startswith("_")
            ):
                assigned_before_install.add(target.attr)

    # Every private *state field* read directly by eventFilter must already exist
    # when installEventFilter is called. Private class methods such as _can_pan()
    # are excluded from the field set. This automatically covers future fields.
    method_names = {
        node.name
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    direct_reads = {
        node.attr
        for node in ast.walk(event_filter)
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr.startswith("_")
            and node.attr not in method_names
        )
    }
    assert direct_reads <= assigned_before_install, (
        f"eventFilter reads fields before init: {sorted(direct_reads - assigned_before_install)}"
    )


def test_event_filter_has_construction_barrier_before_widget_state_access():
    source = _source()
    event_start = source.index("    def eventFilter(self, watched, event):")
    event_end = source.index("    def _stop_pan", event_start)
    block = source[event_start:event_end]

    guard = 'if not getattr(self, "_event_filter_ready", False):'
    assert guard in block
    assert 'return super().eventFilter(watched, event)' in block
    assert block.index(guard) < block.index("selection_mode = self._selection_mode")
    assert block.index(guard) < block.index("self._pixmap")


def test_event_filter_barrier_is_armed_only_after_filter_installation():
    source = _source()
    init_start = source.index("class ImageView(QScrollArea):")
    init_end = source.index("    def _can_pan", init_start)
    block = source[init_start:init_end]
    assert block.index("self._event_filter_ready = False") < block.index("installEventFilter(self)")
    assert block.index("installEventFilter(self)") < block.index("self._event_filter_ready = True")
    assert block.index("self._pixmap") < block.index("installEventFilter(self)")
