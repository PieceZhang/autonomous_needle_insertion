# US / MRI 本地可视化界面

## 本地运行（Conda）
```bash
conda env create -f environment.yml
conda activate us_mri_ui
python app.py
```

启动后浏览器打开：
`http://127.0.0.1:8050`

## 本地运行（pip）
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## 本地部署参数
- 默认只监听本机：`127.0.0.1:8050`
- 自定义端口：`UI_PORT=9000 python app.py`
- 允许局域网访问：`UI_HOST=0.0.0.0 python app.py`
- 数据目录不在项目根目录时：`UI_DATA_ROOT=/your/data/path python app.py`

## 功能
- 2x2 页面：
  - (0,0): US frame
  - (1,0): US + MRI probe plane 融合图
  - (0,1): 3D probe + needle STL 渲染
  - (1,1): MRI 三正交切片
- 根据 `video_info.json` 的 `start_time_ns` + `measured_fps` 计算每帧时间，匹配最近的 probe pose（可能跳过部分视频帧）。
- 使用 `PlusDeviceSet...xml` 中 `Image -> Probe` 变换把 image plane 对齐到 Polaris 坐标。
- 使用 `affine_matrix/registration_result_MRI.npz` 把 Polaris 坐标映射到 MRI 体数据进行重采样。

## 说明
- 当前将 needle 锚定在 probe 轴向附近，以满足“needle 与 probe 在同一轴面插入”的交互基线。
- 若你有更精确的 needle 实时位姿数据，可在 `app.py` 中替换 `needle_T` 为时间同步后的 needle pose。
