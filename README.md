# MeshRev：活塞零件网格逆向建模工具

MeshRev 是一个类似 Geomagic Design X 的网格逆向建模工具，面向活塞发动机零件的 STL/OBJ 扫描数据。技术栈为 Python + PySide6 + PyVista/VTK，CAD 内核 OpenCASCADE（OCP）为可选依赖。

## 安装与运行

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # 核心依赖
pip install -r requirements-cad.txt      # 可选：STEP 导入/导出、实体建模（OCP）
pip install -r requirements-dev.txt      # 可选：测试与代码检查
pip install -e . --no-deps               # 以可编辑模式安装 meshrev 包

python -m meshrev                        # 启动 GUI
python -m meshrev part.stl other.obj     # 启动并导入文件
python -m meshrev --demo                 # 启动并生成示例活塞
python tools/make_demo_piston.py piston.stl --voxel 0.5 --noise 0.01   # 导出示例活塞
```

> 必须用 `python -m meshrev` 启动。`meshrev/io` 只有在 `meshrev/` 目录本身被加入 `sys.path` 时才会遮蔽标准库 `io`。

在无显示器的 Linux 上，Qt 需要 `libegl1`、`libxkbcommon-x11-0` 和 `libxcb-cursor0` 等系统库。GUI 测试用 `xvfb-run` 运行。

## 架构

```
gui  ──▶  io  ──▶  core          （依赖只能向下，core 不导入 Qt）
```

| 包 | 模块 | 职责 |
|---|---|---|
| `core` | `types.py` | `Plane`、`Axis`、`BBox`、`Units`；右手坐标系，Z 轴向上（活塞轴线），单位 mm |
| | `events.py` | `EventBus`：纯 Python 观察者，由 GUI 转发为 Qt Signal |
| | `bodies.py` | `Body` 及其子类：`MeshBody`、`CadBody`、`SectionBody` |
| | `document.py` | `Document`：实体表 + 参数化历史 + 变更事件 |
| | `features/` | `Feature` 基类、`FeatureHistory`（重算/回滚/抑制）、`UndoStack` 与各类命令 |
| | `mesh/` | 三角化、合并重复顶点、显示用分裂法向、统计、抽稀/平滑/补洞 |
| | `section/` | 平面截面（`slice_mesh`），2D 草图实体（`Line2D`/`Arc2D`/`Sketch`） |
| | `cad/` | `CadKernel` 抽象接口，`OccKernel`（OCP）实现：旋转、拉伸、布尔运算、STEP |
| | `analysis/` | 偏差分析占位接口：`compute_deviation` |
| | `samples.py` | 程序化示例活塞（SDF + marching cubes，带环槽、销座、销孔、顶面凹坑） |
| `io` | `registry.py` | 按扩展名注册读写器：`load()` / `save()` / `dialog_filter()` |
| | `mesh_formats.py` | STL（二进制/ASCII）、OBJ、PLY |
| | `step_format.py` | STEP 导入导出（需要 OCP） |
| `gui` | `main_window.py` | 单文档主窗口：菜单、工具栏、停靠面板、拖放导入、最近文件 |
| | `viewport.py` | `Viewport3D`（pyvistaqt）：显示模式、标准视角、鼠标预设、点选 |
| | `scene.py` | `SceneManager`：实体与 VTK actor 的对应、高亮、相机（不依赖 Qt，可离屏测试） |
| | `controller.py` | `DocumentController`：事件桥接、选择集、后台任务、撤销栈 |
| | `feature_tree.py` / `property_panel.py` | 特征历史树；属性面板（根据 `ParamSpec` 自动生成参数编辑器） |

### 核心约定

- **特征是纯函数。** `Feature.execute(ctx)` 只读取输入实体并返回新实体，由 `FeatureHistory` 负责提交。因此耗时特征可以放到后台线程计算，重算结果也是确定的：输出实体的 ID 固定为 `<特征ID>.<序号>`。
- **导入也是特征。** `ImportFeature` 是历史树的根节点，并缓存已加载的数据，重算时不会重新读取大文件。
- **CAD 内核通过 `get_kernel()` 懒加载。** 没有安装 OCP 时，只有 STEP 和实体建模功能不可用。

## 视图操作

| 功能 | 操作 |
|---|---|
| 显示模式 | 线框 F5 / 着色 F6 / 平滑 F7 / 显示网格边 F8 |
| 标准视角 | 前视 Ctrl+1、后视 Ctrl+2、左视 Ctrl+3、右视 Ctrl+4、俯视 Ctrl+5、仰视 Ctrl+6、等轴测 Ctrl+0；适应窗口 F |
| 鼠标（VTK 默认，默认预设） | 左键旋转，中键或 Shift+左键平移，滚轮或右键缩放，左键单击选择 |
| 鼠标（类 Design X） | 右键旋转，Ctrl+右键平移，滚轮或 Shift+右键缩放，左键单击选择 |

- **平滑模式**：顶点法向在二面角大于 30° 的边处拆分，环槽等锐边不会被抹圆。
- **着色模式**：逐面平直着色。
- **投影方式**：默认正交投影。

## 测试

```bash
xvfb-run -a python -m pytest       # 完整测试（含 GUI）
python -m pytest                   # 无显示器时自动跳过 GUI 测试；未安装 OCP 时自动跳过 CAD 测试
ruff check . && ruff format --check .
```
