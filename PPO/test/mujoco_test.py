# sudo apt-get update -qq
# sudo apt-get install -y \
#     libosmesa6-dev \
#     libglx-mesa0 \
#     libglfw3 \
#     libgl1-mesa-dev \
#     libglew-dev \
#     patchelf \
#     glew-utils

    
# pip uninstall cython
# pip install cython==0.29.34


# # Create MuJoCo directory
# mkdir -p ~/.mujoco

# # Download MuJoCo binaries
# wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O ~/.mujoco/mujoco.tar.gz

# # Extract MuJoCo binaries
# tar -zxf ~/.mujoco/mujoco.tar.gz -C ~/.mujoco

# # Clean up tar file
# rm ~/.mujoco/mujoco.tar.gz

# # Add MuJoCo to LD_LIBRARY_PATH
# echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin' >> ~/.bashrc

# # Apply changes
# source ~/.bashrc


# pip install -U 'mujoco-py<2.2,>=2.1'


import mujoco_py
import os
print(os.environ['LD_LIBRARY_PATH'])

mj_path = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)

print(sim.data.qpos)