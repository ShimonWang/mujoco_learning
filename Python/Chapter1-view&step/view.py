import time
import math

import mujoco
import mujoco.viewer

# 从 XML 文件加载模型
m = mujoco.MjModel.from_xml_path('API-MJC/pointer.xml')
# m = mujoco.MjModel.from_xml_path('../../API-MJC/pointer.xml')
# 创建数据结构用于存储仿真状态
d = mujoco.MjData(m)  # MjData:这是存储模拟状态的主要数据结构。它是所有函数读取可修改输入和写入输出的工作空间。

# 启动被动式可视化界面，支持暂停、交互操作，但不控制仿真步进
with mujoco.viewer.launch_passive(m, d) as viewer:
# 这个函数不会阻塞，允许函数继续执行下去。在这种模式下，用户的脚本负责计时和推进物理状态，
# 除非用户显式同步传入的事件，否则鼠标拖拽扰动将不会起作用。
  # Close the viewer automatically after 30 wall-seconds.  # 30 秒后自动关闭查看器。
  start = time.time()  # 返回当前时间戳
  cnt = 0  # 控制输入计数器

  # 只要窗口在运行，并且运行时间小于30s，就持续仿真
  while viewer.is_running() and time.time() - start < 30:  # is_running(): 如果查看器窗口正在运行
    step_start = time.time()

    '''测试step 直接执行一步完整的仿真，包括mj_step1和mj_step2
    mj_step1: 执行位置相关更新，如传感器、控制器等（相当于 prepare）
    mj_step2: 执行动力学更新（相当于执行真正的一步）
    '''
    # d.ctrl[1] = math.sin(cnt)
    # mujoco.mj_step(m, d)
    #mj_step(m: mujoco._structs.MjModel, d: mujoco._structs.MjData, nstep: int = 1) -> None
    #高级模拟，使用控制回调获取外力和控制。可选择重复 nstep 次。
    
    '''测试step1 step2 '''
    mujoco.mj_step1(m, d)
    # d.ctrl[1] = math.sin(cnt)
    # mujoco.mj_step2(m, d)
    #mj_step1(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None
    #分两步高级模拟：在用户设置外力和控制之前
    
    '''测试forward
    mj_forward: 用当前状态（qpos、qvel）计算所有导出量（如力、加速度、传感器等），不推进时间。
    '''
    d.ctrl[0] = math.sin(cnt)  # control
    d.qpos[0] = math.sin(cnt)  # position
    mujoco.mj_forward(m, d)  # 前向动力学带跳过，跳过阶段是mjtStage
    print("qvel:",d.qvel)  # velocity
    print("qacc:",d.qacc)  # acceleration
    print("qpos:",d.qpos)  # position
    
    '''测试inverse
    mj_inverse: 给定加速度、位置、速度，反向计算需要的力（qfrc_inverse）。
    '''
    # d.qacc[0] = math.sin(cnt)
    # d.qpos[0] = 0
    # d.qvel[0] = 0
    # mujoco.mj_inverse(m, d)  # 逆动力学，调用前必须设置qacc
    # print("qfrc_inverse",d.qfrc_inverse)
    
    cnt += 0.01  # 控制输入更新

    # Example modification of a viewer option: toggle contact points every two seconds.
    # 查看器选项的修改示例：每两秒切换一次接触点。
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)  # mjVIS_CONTACTPOINT:contact points
    #lock():作为上下文管理器，为查看器提供互斥锁

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    # 从图形用户界面获取物理状态的变化、应用扰动、更新选项。
    viewer.sync()
    #sync():在 mjModel、mjData 和 GUI 用户输入之间同步状态

    # Rudimentary time keeping, will drift relative to wall clock.
    # 基本计时，相对于挂钟会漂移。
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
