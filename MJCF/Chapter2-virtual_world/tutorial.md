# 仿真世界
## 世界根节点
```xml
<mujoco model="模型名称">
</mujoco>
```
## 仿真计算配置
### compiler节点
```xml
<compiler angle="radian/degree" autolimits="true" >
```
compiler节点中定义的包括angle（角度单位），autolimits（受力限制）等，常用的就像上面这样写就行，规定角度单位为弧度制，受力限制开启。角度单位为弧度制是机器人开发的常用单位制。
### option节点
```xml
<option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" 
density="1.225" viscosity="1.8e-5"/>
```
* timestep代表仿真走一步的时间，也就是运行一次之后仿真计算出timestep时长后的世界，单位秒。timestep是一定要规定的，不然仿真不知道如何计算
* gravity 重力加速度
* wind 风在三个方向的速度
* magnetic="0 -0.5 0"世界磁场，影响磁力传感器
* density 介质密度，水，空气等，单位kg/m³
* viscosity 介质粘度
* integrator= [Euler/RK4/implicit/implicitfast]积分器，默认欧拉，用于仿真世界每步就求解计算，各个优势见下图
* solver= [PGS, CG, Newton]求解器配置，默认牛顿
* iterations="100" 约束求解器最大迭代次数,按需配置。
求解器其他配置：
![](../asset/slove.png)
积分器比较：
![](../asset/integrator.png)
""

## 可视化配置
### visual节点
```xml
<visual>
    <global realtime="1"/>
    <quality shadowsize="16384" numslices="28" offsamples="4" />
    <headlight diffuse="1 1 1" specular="0.5 0.5 0.5" active="1" />
    <rgba fog="0 1 0 1" haze="1 0 0 1"/>
</visual>
```
* global:realtime仿真速度比例，在simulate中可以使用，大于1的按1计算
* quality:画面质量
* headlight：和simulate中自由相机相同方向的光源
* map：鼠标影响操作
* scale：渲染缩放
* rgba: fog 迷雾颜色；haze 地平线颜色

## 资源配置
```xml
<asset>
    <mesh name="tetrahedron" vertex="0 0 0 1 0 0 0 1 0 0 0 1" />
    <mesh file="card.obj" />
    <texture type="2d" file="./king_of_clubs.png" />
    <material name="king_of_clubs" texture="king_of_clubs" />
    <hfield name="agent_eval_gym" file="agent_eval_gym.png" size="10 10 1 1" />
    <texture type="skybox" file="../asset/desert.png"
        gridsize="3 4" gridlayout=".U..LFRB.D.." />
    <texture name="plane" type="2d" builtin="checker" rgb1=".1 .1 .1" rgb2=".9 .9 .9"
        width="512" height="512" mark="cross" markrgb=".8 .8 .8" />
    <material name="plane" reflectance="0.3" texture="plane" texrepeat="1 1" texuniform="true"/>
    <material name="box" rgba="0 0.5 0 1"  emission="0"/>
</asset>
```
### 几何资源 mesh
* vertex 通过坐标点构造几何体
* file 加载obj或者stl模型，不指定name默认为文件名
* hfield通过高度图加载几何体
### 纹理/材质资源
* texture 可以通过加载png对物体进行贴图，使用方法texture->material
* material 材质，可以指定纹理，颜色，反射，发光等
* skybox texture的类型，直接加载即可
**天空盒演示：**
```xml
<texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="0.6 0.8 1" width="256" height="256"/>
```