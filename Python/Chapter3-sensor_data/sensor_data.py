import time
import math

import mujoco
import mujoco.viewer
import cv2
import glfw
import numpy as np

# 从 XML 文件加载模型
# m = mujoco.MjModel.from_xml_path('API-MJC/pointer.xml')
m = mujoco.MjModel.from_xml_path('../../API-MJC/pointer.xml')
# 创建数据结构用于存储仿真状态
d = mujoco.MjData(m)

def get_sensor_data(sensor_name):
    """获取传感器数据"""
    sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    #mj_name2id:获取具有指定mjtObj类型名称的对象的id；mujoco.mjtObj.mjOBJ_SENSOR:sensor
    if sensor_id == -1:
        raise ValueError(f"Sensor '{sensor_name}' not found in model!")
    start_idx = m.sensor_adr[sensor_id]  # sensor_adr:address in sensor array (nsensor x 1)
    dim = m.sensor_dim[sensor_id]  # sensor_dim:number of scalar outputs  (nsensor x 1)
    sensor_values = d.sensordata[start_idx : start_idx + dim]  # sensordata:sensor data array (nsensordata x 1)
    return sensor_values

# 导入glfw，初始化 GUI 窗口
glfw.init()
glfw.window_hint(glfw.VISIBLE,glfw.FALSE)  # 设置窗口不可见（后台offscreen渲染）
window = glfw.create_window(1200,900,"mujoco",None,None)  # 创建窗口
glfw.make_context_current(window)  # 绑定OpenGL上下文到当前窗口

# 创建一个 camera 对象，用于从指定相机视角渲染图像
camera = mujoco.MjvCamera()
camID = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "this_camera")  # 通过相机名称获取 camera id
camera.fixedcamid = camID  # 设置当前camera使用固定视角，相机编号为camID
camera.type = mujoco.mjtCamera.mjCAMERA_FIXED  # 相机类型设为固定相机

# 创建一个场景对象，用于管理可视元素
scene = mujoco.MjvScene(m, maxgeom=1000)  # 场景中最多支持 1000 个几何体
context = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)  # 渲染上下文
mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)  # 设置为 offscreen 渲染，不在 GUI 中直接显示

# 封装图像获取函数：从当前相机视角渲染场景，返回 OpenCV 图像
def get_image(w,h):
    # 定义视口大小
    viewport = mujoco.MjrRect(0, 0, w, h)  # 定义视口大小（窗口区域）
    # 更新场景
    mujoco.mjv_updateScene(
        m, d, mujoco.MjvOption(),  # 使用默认可视化选项
        None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene
    )  # 从camera视角渲染所有对象
    # 渲染到缓冲区
    mujoco.mjr_render(viewport, scene, context)
    # 读取 RGB 数据（格式为 HWC, uint8）
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(rgb, None, viewport, context)
    # 图像上下翻转+转换为OpenCV可读格式（BGR）
    cv_image = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)
    return cv_image


# 启动 MuJoCo 的被动查看器（passive viewer）
with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.  # 30 秒后自动关闭查看器。
  start = time.time()

  # 只要窗口在运行，并且运行时间小于30s，就持续仿真
  while viewer.is_running() and time.time() - start < 30:
    
    # 控制器输入，让某关节维持在2N力或某种控制值
    d.ctrl[1] = 2
    
    step_start = time.time()
    mujoco.mj_step(m, d)  # 执行一步仿真计算（包含动力学、碰撞等）
    
    # 打印某个传感器（角速度）的数据
    sensor_data = get_sensor_data("quat")
    print("sensor_data",sensor_data)
    print(d.sensor("quat").data)
    
    # 渲染并显示摄像头图像
    img = get_image(640,480)
    cv2.imshow("img",img)
    cv2.waitKey(1)

    # Example modification of a viewer option: toggle contact points every two seconds.
    # 每 2 秒切换一次是否显示 contact points
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    # 同步 viewer 与仿真状态
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    # 时间补偿：保持与仿真时间步一致（如 0.002s）
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
