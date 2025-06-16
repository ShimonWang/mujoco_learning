import time
import math

import mujoco
import mujoco.viewer

# 从 XML 文件加载模型
m = mujoco.MjModel.from_xml_path('API-MJC/pointer.xml')
# m = mujoco.MjModel.from_xml_path('../../API-MJC/pointer.xml')
# 创建数据结构用于存储仿真状态
d = mujoco.MjData(m)

# 启动被动式可视化界面，支持暂停、交互操作，但不控制仿真步进
with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.  # 30 秒后自动关闭查看器。
  start = time.time()  # 返回当前时间戳
  cnt = 0  # 控制输入计数器

  # 只要窗口在运行，并且运行时间小于30s，就持续仿真
  while viewer.is_running() and time.time() - start < 30:
    step_start = time.time()
    
    d.ctrl[1] = math.sin(cnt)
    mujoco.mj_step(m, d)  # 执行一步仿真计算（包含动力学、碰撞等）
    
    # print(m.njnt)   # 1
    # print(m.nsite)  # 1
    # print(m.nbody)  # 3
    # print(m.names)
    base_id = mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"pointer")
    # print(f"base_id:{base_id}")
    print(f"d.xpos[base_id]:{d.xpos[base_id]}")  # mjtNum* xpos:刚体坐标系笛卡尔位置
    imu_id = mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,"imu")
    #mj_name2id:获取具有mjtObj类型名称的对象的id;mujoco.mjtObj.mjOBJ_SITE:site 参考点/标记
    print(f"imu_id:{imu_id}")
    print(f"d.site_xpos[imu_id]:{d.site_xpos[imu_id]}")  # site_xpos:笛卡尔site位置
    # w x y z
    # print(d.xquat[base_id])  # mjtNum* xquat:刚体坐标系笛卡尔姿态
      
    cnt += 0.005

    # Example modification of a viewer option: toggle contact points every two seconds.
    # 查看器选项的修改示例：每两秒切换一次接触点。
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    # 从图形用户界面获取物理状态的变化、应用扰动、更新选项。
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    # 基本计时，相对于挂钟会漂移。
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
