import time
import math

import mujoco
import mujoco.viewer
import cv2
import glfw
import numpy as np

# 加载模型和数据对象
# m = mujoco.MjModel.from_xml_path('../../API-MJC/mecanum.xml')
m = mujoco.MjModel.from_xml_path('API-MJC/mecanum.xml')
d = mujoco.MjData(m)

def get_sensor_data(sensor_name):
    """获取传感器数据"""
    sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sensor_id == -1:
        raise ValueError(f"Sensor '{sensor_name}' not found in model!")
    start_idx = m.sensor_adr[sensor_id]
    dim = m.sensor_dim[sensor_id]
    sensor_values = d.sensordata[start_idx : start_idx + dim]
    return sensor_values

# 初始化 GLFW（OpenGL 窗口工具）
glfw.init()
glfw.window_hint(glfw.VISIBLE,glfw.FALSE)
window = glfw.create_window(1200,900,"mujoco",None,None)
glfw.make_context_current(window)

# 创建一个 camera 对象，用于从指定相机视角渲染图像
camera = mujoco.MjvCamera()
camID = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "this_camera")
camera.fixedcamid = camID
camera.type = mujoco.mjtCamera.mjCAMERA_FIXED

# 创建一个场景对象，用于管理可视元素
scene = mujoco.MjvScene(m, maxgeom=1000)
context = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)


# 启动 MuJoCo 的被动查看器（passive viewer）
with mujoco.viewer.launch_passive(m, d) as viewer:
  # 绘制几何体（如球体）
  def draw_geom(type, size, pos, mat, rgba):
    viewer.user_scn.ngeom += 1
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom - 1]   
    mujoco.mjv_initGeom(geom, type, size, pos, mat, rgba)
  
  # 绘制线段
  def draw_line(start, end, width, rgba):
    viewer.user_scn.ngeom += 1
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom - 1]
    size = [0.0, 0.0, 0.0] 
    pos = [0, 0, 0]           
    mat = [0, 0, 0, 0, 0, 0, 0, 0, 0]     
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, size, pos, mat, rgba)
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, width, start, end)
  
  # 绘制箭头
  def draw_arrow(start, end, width, rgba):
    viewer.user_scn.ngeom += 1
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom - 1]
    size = [0.0, 0.0, 0.0] 
    pos = [0, 0, 0]           
    mat = [0, 0, 0, 0, 0, 0, 0, 0, 0]   
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, size, pos, mat, rgba)
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, width, start, end)
  
  cnt = 0
  start = time.time()
  ngeom = viewer.user_scn.ngeom  # 记录初始几何体数量，后续动态追加

  while viewer.is_running() and time.time() - start < 30:
    # 控制信号，驱动轮子运动
    d.ctrl[0] = math.sin(cnt)
    d.ctrl[1] = math.cos(cnt)
    d.ctrl[2] = math.sin(cnt)
    cnt += 0.003
    
    step_start = time.time()
    mujoco.mj_step(m, d)  # 执行一步仿真计算（包含动力学、碰撞等）
    
    '''重置geom数量'''
    viewer.user_scn.ngeom = ngeom
    
    '''3D绘制'''
    size = [0.1, 0.0, 0.0] 
    pos = [0, 0, 1]         
    mat = [1, 0, 0, 0, 1, 0, 0, 0, 1] # 坐标系，空间向量
    rgba = [1.0, 0.0, 0.0, 1.0]     
    draw_geom(mujoco.mjtGeom.mjGEOM_SPHERE, size, pos, mat, rgba)
    
    pos_start = [0, 1, 1]
    end = [0, 3, 4]
    draw_line(pos_start, end, 20, rgba)
    
    pos_start2 = [0, 0, 1]
    end2 = [0, -1, 1]
    rgba2 = [0.0, 1.0, 0.0, 0.5]
    draw_arrow(pos_start2, end2, 0.1, rgba2)
    
    '''速度跟踪'''
    base_lin_vel = get_sensor_data("base_lin_vel")
    base_pos = get_sensor_data("base_pos")
    base_pos[2] += 1
    lin_start = base_pos
    lin_end = lin_start + base_lin_vel * 2
    draw_arrow(lin_start, lin_end, 0.1, rgba2)
    
    w=640
    h=480
    viewport = mujoco.MjrRect(0, 0, w, h)
    # 更新场景
    mujoco.mjv_updateScene(
        m, d, mujoco.MjvOption(), 
        None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene
    )
    # 渲染到缓冲区
    mujoco.mjr_render(viewport, scene, context)
    
    '''2D绘制'''
    mujoco.mjr_text(mujoco.mjtFont.mjFONT_NORMAL, "Albusgive", context, 0, 0.8, 1, 0, 1)
    viewport2 = mujoco.MjrRect(50, 100, 50, 50)
    mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, "github", "Albusgive", context)
    mujoco.mjr_rectangle(viewport2, 0.5, 0, 1, 0.6)
    viewport3 = mujoco.MjrRect(100, 200, 150, 50)
    mujoco.mjr_label(viewport3, mujoco.mjtFont.mjFONT_NORMAL, "Albusgive", 0, 1, 1, 1, 0, 0,0, context)
    
    # 读取 RGB 数据（格式为 HWC, uint8）
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(rgb, None, viewport, context)
    cv_image = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)
    
    cv2.imshow("img",cv_image)
    cv2.waitKey(1)


    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
