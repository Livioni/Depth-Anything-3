# Tools

这个目录包含项目中使用的一些工具脚本。

## extract_video_frames.py

视频帧提取脚本，用于从MP4视频文件中提取每一帧并保存为PNG图像。

### 使用方法

```bash
python tools/extract_video_frames.py
```

### 功能

- 从指定的MP4视频文件中提取每一帧
- 将帧保存为PNG格式的RGB图像
- 文件命名格式：000000.png, 000001.png, 000002.png...
- 保存到指定输出目录

### 当前配置

脚本当前配置为处理以下文件：
- 输入视频：`datasets/realm/2026_02_03_07_42_05_pi0_rollout_put_green_block_in_bowl_V-AUG_0/rgb_wrist_camera.mp4`
- 输出目录：`datasets/realm/2026_02_03_07_42_05_pi0_rollout_put_green_block_in_bowl_V-AUG_0/images/`

### 自定义使用

如果需要处理其他视频文件，可以修改脚本中的 `main()` 函数中的路径配置。

### 依赖

- opencv-python (已在requirements.txt中)
- pillow (PIL) (已在requirements.txt中)
- numpy (已在requirements.txt中)

## create_pointcloud.py

点云生成脚本，用于从RGB图像、深度图和相机内参生成彩色点云。

### 使用方法

```bash
python tools/create_pointcloud.py
```

### 功能

- 从RGB图像和深度图读取数据
- 使用相机内参将像素坐标转换为3D世界坐标
- 生成包含位置和颜色信息的点云
- 支持下采样以减少点云密度
- 保存为PLY格式的点云文件

### 当前配置

脚本当前配置为处理以下文件：
- RGB图像：`datasets/realm/2026_02_03_07_42_05_pi0_rollout_put_green_block_in_bowl_V-AUG_0/images/000000.png`
- 深度图：`datasets/realm/2026_02_03_07_42_05_pi0_rollout_put_green_block_in_bowl_V-AUG_0/depth_wrist_camera/000000.png`
- 相机参数：`datasets/realm/2026_02_03_07_42_05_pi0_rollout_put_green_block_in_bowl_V-AUG_0/camera_params.json`
- 输出文件：`pointcloud_000000.ply`

### 输出说明

- 生成的PLY文件包含每个点的XYZ坐标（米为单位）和RGB颜色
- 使用下采样因子2，减少计算量和文件大小
- 深度值从毫米转换为米
- 自动过滤无效深度值

### 自定义使用

可以通过修改脚本中的参数来自定义：
- `frame_idx`: 指定使用哪一帧的相机参数
- `downsample_factor`: 控制点云密度（1=全分辨率，2=1/4密度）
- 文件路径：修改输入输出路径

### 依赖

- opencv-python (已在requirements.txt中)
- pillow (PIL) (已在requirements.txt中)
- numpy (已在requirements.txt中)

## verify_pointcloud.py

点云质量验证脚本，用于分析生成的点云质量和统计信息。

### 使用方法

```bash
python tools/verify_pointcloud.py
```

### 功能

- 读取PLY格式的点云文件
- 分析点云的空间分布和颜色统计
- 生成可视化图表（XY/XZ投影和深度分布直方图）
- 输出详细的质量统计信息

### 输出内容

1. **基本信息**: 总点数
2. **坐标范围统计**: X/Y/Z轴的最小值、最大值和范围
3. **距离统计**: 到相机的距离分布（最小值、最大值、平均值、中位数）
4. **颜色统计**: RGB三个通道的范围和平均值
5. **可视化图表**: 保存为`pointcloud_analysis.png`

### 依赖

- numpy (已在requirements.txt中)
- matplotlib (需安装: `pip install matplotlib`)

## extract_coco_masks.py

COCO格式mask提取脚本，用于从COCO格式的JSON文件中提取二值mask并保存为PNG格式。

### 使用方法

```bash
python tools/extract_coco_masks.py input_json_file.json --output_dir output_directory
```

### 功能

- 读取COCO格式的RLE编码mask数据
- 支持字母数字混合编码的RLE字符串解码
- 将每一帧mask保存为独立的PNG文件
- PNG格式对二值图像具有极佳的压缩效果

### 参数说明

- `input_json_file`: 输入的COCO格式JSON文件路径
- `--output_dir`: 输出目录，默认为 `extracted_masks`
- `--prefix`: 输出文件名前缀，默认为空

### 示例

```bash
# 基本用法
python tools/extract_coco_masks.py datasets/droid/Fri_Jul__7_09:45:39_2023/18026681/robot_masks/tracked_masks_coco.json

# 指定输出目录和前缀
python tools/extract_coco_masks.py datasets/droid/Fri_Jul__7_09:45:39_2023/18026681/robot_masks/tracked_masks_coco.json --output_dir robot_masks --prefix robot_
```

### 输出格式

- 每个mask保存为独立的PNG文件
- 文件名格式: `{prefix}{frame_index:06d}.png`
- 图像尺寸与原始mask相同（例如1280x720）
- 二值图像：0=背景，255=前景

### 存储效率

PNG格式对二值图像的压缩效果非常好：
- 1280x720的二值mask通常只有1-3KB
- 比其他格式节省大量存储空间
- 无损压缩，保持精确的二值信息

### 依赖

- pillow (PIL) (已在requirements.txt中)
- numpy (已在requirements.txt中)