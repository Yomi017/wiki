---
{"dg-publish":true,"permalink":"/notion/practical-knowledge/computer-science/software/unreal-engine-5-4/third-party-plugin/colosseum/"}
---

**官方文档：**[Home - Colosseum](https://codexlabsllc.github.io/Colosseum/)

**官方介绍：**

Colosseum is a simulator for robotic, autonomous systems, built on [Unreal Engine](https://www.unrealengine.com/) (we now also have an experimental [Unity](https://unity3d.com/) release). It is open-source, cross platform, and supports software-in-the-loop simulation with popular flight controllers such as PX4 & ArduPilot and hardware-in-loop with PX4 for physically and visually realistic simulations. It is developed as an Unreal plugin that can simply be dropped into any Unreal environment. Similarly, we have an experimental release for a Unity plugin.

This is a fork of the AirSim repository, which Microsoft decided to shutdown in July of 2022. This fork serves as a waypoint to building a new and better simulation platform. The creater and maintainer of this fork is Codex Laboratories LLC (our website is [here](https://www.codex-labs-llc.com/)). Colosseum is one of the underlying simulation systems that we use in our product, the [SWARM Simulation Platform](https://www.swarmsim.io/). This platform exists to provide pre-built tools and low-code/no-code autonomy solutions. Please feel free to check this platform out and reach out if interested.

Colosseum是一款专为机器人和自主系统打造的模拟器，它基于虚幻引擎（Unreal Engine）构建（我们现在也有一个实验性的 Unity 版本）。它开源、跨平台，并支持与 PX4 和 ArduPilot 等流行飞行控制器进行软件在环仿真，以及与 PX4 进行硬件在环仿真，以实现物理上和视觉上逼真的模拟效果。它被开发为一个虚幻引擎插件，可以轻松集成到任何虚幻引擎环境中。类似地，我们也有一个实验性的 Unity 插件版本。

这是 AirSim 代码库的一个分支，微软已于 2022 年 7 月决定停止该项目。这个分支是构建一个全新且更优的模拟平台过程中的一个重要节点。该分支的创建者和维护者是 Codex Laboratories LLC（我们的网站在此处）。科洛西姆是我们产品——SWARM 模拟平台（SWARM Simulation Platform）——中使用的底层模拟系统之一。该平台旨在提供预构建工具以及低代码/无代码的自主解决方案。欢迎体验该平台，如果您感兴趣，请随时与我们联系。

## 1. 安装教程

### ① 准备

UE 5.4， 一个你想导入的项目 (或者直接新建一个C++项目)，VS 2022

  

**VS 2022:**

**VS的选择非常重要，较新版本中的 MSVC 和 Windows SDK 可能会导致编译失败，如：**

**MSVC 14.4x+ 中的宏被删除：[Error C4668 : 没有将“__has_feature”定义为预处理器宏](https://schizo.top/archives/368)**

可行的尝试：安装[VS 2022 17.8.7](https://learn.microsoft.com/en-us/visualstudio/releases/2022/release-history#fixed-version-bootstrappers)，在 **Visual Studio Installer** 中选择对应版本上的**修改**，在**工作负荷**选择**桌面应用和移动应用**的**使用 C++ 的桌面开发**，在**单个组件**搜索并选择：

**.NET Framework 4.6.2 SDK**

**C++ CMake tools for Windows (用于 Windows 的 C++ CMake 工具)**

**MSVC v143 -VS 2022 C++ x64/x86 生成工具 (v14.38-17.8)**

**Windows 10 SDK (10.0.20348.0)**

  

**导入的项目 (或者新建的C++项目)：**

新建的C++项目如果出现编译问题请查看日志，大概率也是上面的VS问题。

导入的项目如果是个**蓝图项目**，需要先进入 **UE 5.4** 选择 **Tools (工具)** ，选择**新建 C++ 类**，这里随便新建，正常来说选**空**就行，然后 UE 5.4 就会自动生成项目，在是否选择编译的时候选择不用，关闭UE 5.4，这时项目的根目录应该会生成一个`.sln`

  

**Colosseum 安装**

选择一个文件夹进行源码克隆 (最好不在C盘下) ，点击win键，搜索 **Developer Command Prompt for VS 2022 LTSC** ，直接打开，输入：

```PowerShell
git clone https://github.com/CodexLabsLLC/Colosseum.git 

# 如果速度过慢可以浅克隆
# git clone https://github.com/CodexLabsLLC/Colosseum.git --depth 1

# 防止 Eigen 没装
cd Colosseum
git submodule update --init --recursive

# 完成后
build.cmd
```

如果 `build.cmd` 出现一些找不到 **VS 2022** 的错误，可以选择：

重装 **VS 2022** ；进入**VS Installer** 选择**更多**，再选择**修复**；重启电脑

这些操作试过之后应该能解决。

  

**导入插件：**

进入 **Colosseum/Unreal** ，将 **Plugins** 文件夹复制到**你项目的根目录下 (和 Config 和 Content 同层级那个)** 。

[修改配置文件](https://www.bilibili.com/opus/1003021395027820544) **[%appdata%\Unreal Engine\UnrealBuildTool\BuildConfiguration.xml](https://www.bilibili.com/opus/1003021395027820544)** ，改为：

```XML
<?xml version="1.0" encoding="utf-8" ?>

<Configuration xmlns="https://www.unrealengine.com/BuildConfiguration">

<WindowsPlatform>

<CompilerVersion>14.38.33130</CompilerVersion>

<ToolchainVersion>14.38.33130</ToolchainVersion>

</WindowsPlatform>

</Configuration>
```

重新回到项目根目录，右键`.sln` 文件，使用 **VS 2022** 打开，上方选择 **Development Editor** 和 **Win64**，找到侧边的**解决方案资源管理器**，能找到一个你项目名字的文件夹在 **Game** 文件夹下面

**(注意：没有后缀)**，右键选择**设为启动项目**，最后 **Ctrl + Shift + B** 编译项目即可。