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
| | `bodies.py` | `Body` 及其子类：`MeshBody`、`CadBody`、`SectionBody`、`RegionSetBody`（分割结果）、`DatumAxisBody`、`DatumPlaneBody` |
| | `document.py` | `Document`：实体表 + 参数化历史 + 变更事件 |
| | `features/` | `Feature` 基类、`FeatureHistory`（重算/回滚/抑制）、`UndoStack` 与各类命令；`recognition.py`：自动分割、RANSAC、基准轴/基准平面特征 |
| | `mesh/` | 三角化、合并重复顶点、显示用分裂法向、统计、抽稀/平滑/补洞 |
| | `section/` | 平面截面（`slice_mesh`），2D 草图实体（`Line2D`/`Arc2D`/`Sketch`） |
| | `cad/` | `CadKernel` 抽象接口，`OccKernel`（OCP）实现：旋转、拉伸、布尔运算、STEP |
| | `analysis/` | 偏差分析占位接口：`compute_deviation` |
| | `primitives.py` | **基元识别**：法向估计、最小二乘/RANSAC 平面与圆柱提取、自动分割（Auto Segment） |
| | `mesh/topology.py` | `MeshGeometry`：向量化的面邻接、二面角、连通分量、面/顶点法向 |
| | `mesh/curvature.py` | 逐面主曲率估计（normal cycle 边张量法，排除锐边） |
| | `samples.py` | 程序化示例活塞（SDF + marching cubes，带环槽、销座、销孔、顶面凹坑）、带端盖圆柱 |
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

## 基元识别（`meshrev/core/primitives.py`）

活塞 80% 以上的表面由平面和圆柱面构成：
- 平面：顶面、环槽侧面、销座内侧面、底面
- 圆柱面：裙部、环岸、环槽底、销孔、衬套孔

识别分为四层，每层都可以单独调用：

| 层 | 接口 | 算法 |
|---|---|---|
| 最小二乘拟合 | `fit_plane`、`fit_cylinder`、`fit_sphere`、`fit_mesh_faces` | **平面**：加权 PCA，得到 `ax+by+cz+d=0`（单位法向）。<br>**圆柱**：先由法向协方差最小特征向量得到轴线，再做代数圆拟合作为初值，最后用 5 参数 Levenberg–Marquardt 最小化 `|到轴距离 − r|`（解析雅可比，可选 soft-L1 抗差）。<br>网格上在区域**顶点**上拟合，面心有弦高误差；同时判定孔/轴（`concave`）。 |
| 法向估计 | `estimate_normals`，`MeshGeometry.face_normals` / `vertex_normals` | 点云用 kNN PCA 并统一朝外；网格直接使用面法向，以及按面积加权的顶点法向。 |
| RANSAC | `ransac_plane`、`ransac_cylinder`、`detect_primitives`、`extract_planes`、`extract_cylinders`、`merge_coaxial_cylinders` | 参照 Schnabel 2007：<br>• 最小样本：平面用 3 点并校验法向；圆柱用 2 个带法向的点（`a = n₀×n₁`，法线交点为圆心）。<br>• 局部采样；按距离与法向夹角双重判定内点；保留最大连通分量；最后最小二乘精化。<br>• 平面与圆柱同时竞争，避免圆柱上的窄条被误判为平面。<br>• 同轴圆柱（两侧销座孔）自动合并后重新拟合。 |
| 自动分割 | `auto_segment` → `SegmentationResult` | ① 区域生长：锐边（二面角大于 `sharp_angle_deg`）断开，主曲率跳变也断开。<br>② 逐区域做稳健拟合并分类为平面、圆柱、球面或自由曲面；只有当曲面模型把 RMS 降低一半以上时，才会替代平面。<br>③ 用 RANSAC 拆分大的自由曲面区域，例如经相切圆角连成一片的环槽区。<br>④ 基元区域向相邻的自由曲面面片生长，使边界干净。<br>⑤ 合并共面、同轴的相邻区域，把碎片并入邻区，最后重新拟合。 |

```python
from meshrev import io as mio
from meshrev.core.primitives import auto_segment, extract_cylinders, fit_mesh_faces, PrimitiveType

mesh = mio.load("piston.stl")[0].polydata
seg = auto_segment(mesh)  # SegmentationResult
for region in seg.regions[:5]:
    print(region.id, region.type.label, region.fit.describe() if region.fit else "")

holes = [c for c in extract_cylinders(mesh) if c.primitive.concave]
pin_bore = min(holes, key=lambda c: c.primitive.radius)  # 内壁 r≈36，销孔 r≈11
print(pin_bore.primitive.axis, pin_bore.primitive.radius)  # 两侧销座孔已合并拟合
```

### 精度与性能

以下为本仓库测试和示例活塞的实测值：

**合成数据（120° 圆弧，σ = 0.005–0.01 mm）**
- 圆柱轴线误差小于 0.02°（实测约 0.006°）
- 半径误差小于 2 µm

**示例活塞（0.8 mm 体素，约 18 万面）**
- 自动分割能识别：顶面、底面、顶面下方、两个销座内侧面；3 条环槽的 6 个侧面和 3 个槽底；环岸；裙部；两侧内壁；两个销孔；球面凹坑。
- 用两侧销孔区域合并拟合基准轴：轴线方向误差约 0.002°，到理论轴线的距离小于 1 µm，半径 11.001 mm。

**分割耗时（本容器单线程）**

| 网格规模 | 耗时 |
|---|---|
| 18 万面 | 约 3.6 s |
| 45 万面 | 约 11 s |

GUI 中分割在后台线程运行，并显示进度。

**已知限制**
- 当环槽宽度仅约 1.5 个网格边长时，其边界区域可能成为小的自由曲面或圆角区域。
- 带噪声数据上，个别环岸可能被分成 2–3 段同半径圆柱。

### GUI 工作流

1. 导入网格，或选择“文件 → 打开示例活塞”。
2. 执行“识别 → 自动分割”（Ctrl+Shift+A）。区域按颜色显示，可在“识别”菜单中切换“按区域着色”或“按基元类型着色”；原网格会自动隐藏。
3. 在视口中左键单击区域，Ctrl+单击可多选；也可以在特征树的“平面 / 圆柱面 …”分组中选择。选中的区域以黄色高亮，属性面板显示方程、法向、轴线、半径和 RMS。选择分组节点会高亮该类型的全部区域。
4. 执行“识别 → 创建基准轴”（Ctrl+Shift+X）或“创建基准平面”（Ctrl+Shift+P），会生成参数化特征：
   - **基准轴**：视口中显示为橙色轴线，标签为 `A1`、`A2`……
   - **基准平面**：显示为半透明平面，标签为 `P1`……
5. 在属性面板中修改分割参数（锐边角度、曲率容差、拟合公差等），点击“应用并重新生成”后，下游的基准特征会自动重算。
6. 执行“识别 → RANSAC 基元提取”（Ctrl+Shift+R），可以一次性提取平面和圆柱，并为圆柱生成轴线。

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
