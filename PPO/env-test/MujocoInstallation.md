
---

### 步骤 1：更新软件包列表并安装依赖库
首先更新您的系统的软件包列表，并安装MuJoCo运行所需的依赖库。打开终端并运行以下命令：

```bash
sudo apt-get update -qq
sudo apt-get install -y \
    libosmesa6-dev \
    libglx-mesa0 \
    libglfw3 \
    libgl1-mesa-dev \
    libglew-dev \
    patchelf \
    glew-utils
```

### 步骤 2：安装Cython（解决兼容问题）
MuJoCo-py需要特定版本的Cython，因此我们需要先卸载当前的Cython版本并安装指定版本。使用以下命令：

```bash
pip uninstall cython
pip install cython==0.29.34
```

### 步骤 3：创建MuJoCo目录
为了存储MuJoCo文件，我们需要创建一个目录。运行以下命令：

```bash
mkdir -p ~/.mujoco
```

### 步骤 4：下载MuJoCo二进制文件
从MuJoCo官方网站下载适用于Linux的二进制文件。使用`wget`命令：

```bash
wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O ~/.mujoco/mujoco.tar.gz
```

### 步骤 5：解压MuJoCo二进制文件
解压下载的MuJoCo文件到创建的MuJoCo目录中：

```bash
tar -zxf ~/.mujoco/mujoco.tar.gz -C ~/.mujoco
```

### 步骤 6：清理压缩包
解压完成后，可以删除压缩包文件：

```bash
rm ~/.mujoco/mujoco.tar.gz
```

### 步骤 7：配置环境变量
将MuJoCo的bin目录添加到系统的`LD_LIBRARY_PATH`环境变量中。编辑`~/.bashrc`文件，添加以下行：

```bash
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin' >> ~/.bashrc
```

然后执行以下命令以使更改生效：

```bash
source ~/.bashrc
```

### 注意：在 Colab 中的环境变量设置
在 Colab 环境中，使用 `!source ~/.bashrc` 无法使环境变量持久生效，因为每个 shell 命令都在新的 shell 实例中运行，`.bashrc` 文件的修改不会持续。

因此，建议在 Python 中直接通过 `os.environ` 设置环境变量。以下是设置环境变量的代码示例：

```python
import os

# 设置MuJoCo的bin目录到LD_LIBRARY_PATH
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ":/root/.mujoco/mujoco210/bin"
```

### 步骤 8：安装MuJoCo-py
使用pip安装MuJoCo-py，这是MuJoCo的Python接口。为了避免安装最新版本的MuJoCo-py（它可能与MuJoCo 2.1不兼容），我们安装一个指定版本的MuJoCo-py：

```bash
pip install -U 'mujoco-py<2.2,>=2.1'
```

### 步骤 9：验证安装
安装完成后，您可以运行以下Python代码来验证MuJoCo是否成功安装并配置：

```python
import mujoco_py
import os

# 输出当前环境变量LD_LIBRARY_PATH
print(os.environ['LD_LIBRARY_PATH'])

# 自动发现MuJoCo路径
mj_path = mujoco_py.utils.discover_mujoco()

# 设置MuJoCo模型文件路径
xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')

# 加载模型
model = mujoco_py.load_model_from_path(xml_path)

# 初始化仿真环境
sim = mujoco_py.MjSim(model)

# 打印模型的初始位置（qpos）
print(sim.data.qpos)
```

如果您能成功看到模型的初始位置输出，说明安装和配置已经成功。

---

### 额外说明：
1. **MuJoCo版本**：在此指南中，我们使用的是MuJoCo 2.1版本。请根据实际需求选择合适版本。
2. **可能的错误**：如果您在安装过程中遇到任何问题，请确保您的操作系统版本与依赖库的版本兼容。遇到问题时，您可以检查MuJoCo的官方文档或社区论坛获取帮助。