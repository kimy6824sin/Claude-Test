# MeshRev：活塞零件网格逆向建模工具

> 开发规范见 [`Claude.md`](Claude.md)。仓库根目录的 `3.stl`–`6.stl` 是真实样件（进气法兰 ×2、飞轮齿圈、活塞），被用作测试和演示案例。

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
| | `section/` | 精确平面截面 `slice_mesh`（边键拓扑串接、缺口桥接）；草图实体 `Line2D` / `Arc2D` / `Circle2D` / `Sketch`（外轮廓 + 孔） |
| | `sketch_fit.py` | **网格草图拟合**：降噪排序、直线/圆弧自动分段、LSQ + RANSAC 拟合、切点/交点求顶点、孔位层级 |
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

## 网格草图（Mesh Sketch）

对应 Design X 的“网格草图”：用平面截取网格，把截面拟合成结构化的 2D 草图，直接交给 `CadKernel.extrude` / `revolve` 使用。

**草图平面**可以是：
- XY / YZ / ZX 平面，过网格中心，可加偏移；
- 所选**基准平面**；
- 所选**基准轴**：平面包含该轴，可绕轴旋转任意角度，草图 x 轴沿轴线方向，适合回转件（如活塞）的旋转轮廓。

```python
from meshrev.core.section import slice_mesh
from meshrev.core.sketch_fit import fit_section
from meshrev.core.features.sketching import plane_through_axis

curve = slice_mesh(mesh, plane_through_axis(pin_axis, angle_deg=0))  # 3D 截面折线
result = fit_section(curve)  # SketchFitResult
for line in result.describe():
    print(line)  # Line((x1, y1), (x2, y2)) / Arc(center, r, start, end, ccw) / Circle(...)
sketches = result.to_sketches()  # 每个材料区域一个草图：外轮廓 + 孔
solid = get_kernel().extrude(sketches[0], sketches[0].plane.normal, 10.0)
```

### 切片算法（`core/section/slicer.py`）

1. **符号扰动**：计算顶点到平面的有符号距离；恰好落在平面上的顶点视为 `+ε`，因此每个被切三角形恰好有两条相交边，彻底避免“平面过顶点、过棱、过面”的退化情况。
2. **边键求交**：每条相交边按排序后的顶点对编码成键，交点 `p = pa + t(pb − pa)`，其中 `t = sa/(sa − sb)`。共享同一条边的两个三角形得到的是**同一个节点**，所以串接是拓扑精确的，不依赖距离容差。
3. **图遍历**：每个被切三角形是连接两个节点的一条边。从度为 1 的节点出发得到开放链（网格边界、破洞），剩余的环就是闭合轮廓。
4. **缺口桥接**：端点距离小于 `gap_tolerance`（默认为中位段长的 3 倍）的开放链按最近端点贪心连接；链的首尾相接则闭合；超出容差的缺口保持开放并在结果中报告，不做臆测。

### 2D 拟合（`core/sketch_fit.py`）

| 步骤 | 做法 |
|---|---|
| 降噪与排序 | 截面点已由切片按拓扑排好序。先删除孤立的往返尖刺（偏离邻点弦的距离大于 3×tol，且大于弦长）；再加密或抽稀到均匀间距，同时**保留全部原始顶点**，真实尖角不会被抹掉。 |
| 公差 | 默认取 `max(4σ, 截面尺寸 × 0.1%)`。σ 由原始顶点到邻点弦的距离，按 MAD 稳健估计。这个下限可以避免把网格面片棱角和扫描噪声当成特征。 |
| 分段 | 转角大于 35° 的尖角是硬断点。两个断点之间用**贪心最长基元**（倍增探测 + 二分）选择直线或圆弧，圆弧的拱高必须不小于公差。然后移动断点，使两侧总平方误差最小，以处理没有尖角的相切过渡。最后合并共线的直线和共圆的圆弧。 |
| 拟合 | 直线用总体最小二乘（PCA，可处理竖直线）；圆先做 Kasa 代数拟合，再用解析雅可比的 LM 做几何精化。LSQ 超出公差时改用 RANSAC 一致集（直线两点、圆三点外接圆）。最终拟合**只用原始测量顶点**，插值点位于弦上，会使圆偏小。 |
| 顶点 | 相邻图元共用一个精确顶点：清晰相交（夹角大于 8°）时取交点；相切过渡时取切点（圆心到直线的垂足，或两圆连心线上的点）。圆弧按“起点 → 拟合中点 → 终点”三点重建，所以轮廓闭合达到机器精度。 |
| 设计意图（Claude.md #5） | • 与草图坐标轴夹角小于 0.5° 的直线吸附为严格水平或竖直。<br>• 短于 5×tol 且两侧明显相交的小图元（倒角、扫描圆角、体素台阶）折叠为尖角。<br>• 拱高小于公差的圆弧改为直线。<br>• 足够圆的闭合轮廓识别为整圆（螺栓孔、螺纹孔）：RMS/r ≤ 3%，或误差在公差 + 网格弦高以内。六边形不会被误判为圆。 |

### 复杂断开轮廓（孔壁截面等）的处理策略

以过销孔轴线截活塞为例，截面会被销孔断成多个互不相连的区域（顶部环带、两侧销座、裙部）。凸台、螺纹孔、铸造孔和扫描破洞还会产生嵌套或开放的轮廓。处理方式：

1. **拓扑串接，不做距离猜测**：边键保证同一条边只产生一个节点，相邻轮廓即使相距 0.01 mm 也不会被误连。
2. **分类**：闭合环、开放链（网格破洞、截面刚好擦过边界）和碎环分开处理。碎环的周长小于 20×tol（默认），当作噪声丢弃。
3. **缺口桥接**：只连接端点足够近的开放链；剩下的开放链作为开放轮廓拟合，并在结果里标记，GUI 中显示为“开放”，不参与建实体。
4. **包含层级**：用偶奇规则射线法计算每个闭合环的包含深度。深度 0 为外轮廓，1 为孔（销孔壁、螺栓孔），2 为孔中的岛（凸台），依此类推。统一方向：材料边界逆时针，孔顺时针。
5. **按区域组装草图**：`to_sketches()` 为每个偶数深度的环生成一个 `Sketch`，其内含的直接子环（奇数深度）作为孔。OCC 端据此构造带孔平面面，孔的线框按方向自动反向，可以直接拉伸或旋转。
6. **每个环独立拟合**：孔整体识别为 `Circle`，外轮廓分段拟合，各环的公差和偏差分别报告。

实测：
- **3.stl 法兰截面**：4 条切线 + 4 段圆弧；Ø27 通孔 r = 13.494；两个螺栓孔 r ≈ 3.10 / 3.05。拉伸后的体积与原始截面面积相差 0.07%。
- **6.stl 真实活塞**：沿 RANSAC 拟合的裙部轴线（r = 32.93）做轴向截面，得到 40 条直线 + 41 段圆弧的闭合轮廓，最大偏差不超过 2.5 倍公差。
- 单次截面 + 拟合通常在 0.1–0.5 s 内完成。

GUI：选中基准平面或基准轴（可选），执行“识别 → 网格草图…”（Ctrl+Shift+K），设置平面、偏移、旋转角度和公差。草图会叠加显示在网格之上：直线蓝色、圆弧洋红、整圆橙色、顶点黑点，视角自动正视草图平面。属性面板逐条列出 `Line` / `Arc` / `Circle` 及其偏差；修改参数后会重新生成。

## B-Rep 实体建模（OpenCASCADE）

### 内核选型

| 方案 | 安装 | 结论 |
|---|---|---|
| **OCP（`cadquery-ocp`）** | `pip install cadquery-ocp`，Win / Linux / macOS 均有 wheel，与 venv / pip 工作流一致 | **采用**。它就是 CadQuery 和 build123d 所用的 OCCT 绑定，覆盖完整的 OCCT 7.7+/8.x API，维护活跃 |
| build123d | pip 可装，依赖 OCP | 可选。其 API 更 Pythonic，但我们已有自己的特征层与 `CadKernel` 抽象；它的 `Shape.wrapped` 就是 `TopoDS_Shape`，可以直接互通 |
| pythonocc-core | 基本只能通过 conda 安装 | 不采用：与 pip/venv 混用麻烦，发布周期较慢 |

所有建模都通过 `core/cad/kernel.py` 中的 `CadKernel` 接口完成，OCP 的具体实现在 `occ_kernel.py`。因此内核可替换、可选安装；未安装 OCP 时，网格功能照常可用。

### 建模特征（`core/features/modeling.py`，全部参数化，可撤销）

| 特征 | 输入 | 参数 | 说明 |
|---|---|---|---|
| `ExtrudeFeature` 拉伸 | 网格草图 | 高度、方向（法向 / 反向 / 对称）、区域 | 外轮廓 + 孔 → 带孔平面 → `BRepPrimAPI_MakePrism` |
| `RevolveFeature` 旋转 | 过基准轴的网格草图 | 角度（默认 360°）、半截面侧 | 截面在轴线处切开（偶奇配对闭合），取所选一侧重新拟合，得到半截面，再绕草图 u 轴旋转；同时输出半截面草图 |
| `CylinderFeature` 圆柱 | 基准轴 | 半径（0 = 拟合半径）、长度（0 = 贯穿全部）、轴向偏移 | 销孔、螺栓孔、凸台等的工具体 |
| `BooleanFeature` 布尔 | 目标、工具 | 求差 / 求并 / 求交 | 结果体标记 `consumes`，界面中隐藏被消耗的两个操作体；撤销后恢复显示 |

活塞典型建模链：

```
网格 → RANSAC 基准轴（裙部 r=43、销孔 r=11）
     → 网格草图（过裙部轴线、90°，避开销孔） → 旋转 360°，得到带环槽和燃烧室凹坑的回转体
     → 圆柱（销孔基准轴，贯穿） → 布尔求差 → 导出 STEP / IGES
```

示例活塞实测：
- 旋转体为 1 个有效实体（BRepCheck 通过），体积 162 762 mm³，与解析值相差不到 1%；
- 销孔求差去除体积与理论值（2πr²·壁厚）相差不到 8%；
- 把销孔半径改为 12 mm 后自动重建，去除体积按 r² 比例增加；
- STEP 和 IGES 往返后体积误差 < 1e-6，仍为实体。

> 注意：旋转截面不要穿过销孔轴线。选 90° 截面可以得到完整的回转体；若截面经过销孔，半截面会被销孔断开，旋转后成为两个实体（沟槽环）。

### 导出

“建模 → 导出 CAD（STEP / IGES）…”，或“文件 → 导出”并选择 `.step` / `.stp` / `.iges` / `.igs`。
- **STEP**：AP214，单位 mm；
- **IGES**：BRep 模式（186 流形实体），实体不会退化为曲面。
- 多个实体合并为一个 compound 导出。

### 参数面板

选中特征树中的特征，属性面板会自动生成对应的参数编辑器（数值框、下拉框、复选框）：
- 点“应用并重新生成”，或勾选 **“实时重建”**（参数停顿 0.4 s 后自动应用）；
- 下游特征会自动重算，例如修改圆柱半径后，布尔结果随之更新；
- 每次修改都进入撤销栈。

菜单：“建模”下有拉伸（Ctrl+Shift+E）、旋转（Ctrl+Shift+V）、圆柱（由基准轴）、由基准轴挖孔…、布尔运算…（Ctrl+Shift+B）和导出 CAD。

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
