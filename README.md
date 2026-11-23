# MHRise for Android

在 64 位 Android 设备上运行自己持有的 Switch 版《怪物猎人 崛起》16.0.1。启动器从 ExeFS 加载游戏的 ARM64 代码，游戏资源从 RomFS 读取。图形后端将 NVN 命令映射到 Vulkan，并在设备上把 Maxwell 着色器翻译成 SPIR-V。

这个分支整合了 Lua MOD、触屏操作和中文版设置菜单，并修正了命中时血花连成大片、部分碰撞粒子异常放大的渲染问题。

## 免责声明

本项目由社区独立维护，与卡普空（CAPCOM）及其旗下游戏的官方开发或发行团队无隶属或合作关系，也未获其赞助或背书。《怪物猎人 崛起》、其 DLC 及相关商标的权利归各自权利人所有。

本仓库不包含或分发原版游戏的源代码、可执行文件、ROM 镜像、游戏资源包、Switch 解密密钥或玩家存档。运行所需的游戏程序（ExeFS）和游戏资源（RomFS）须由玩家从自己合法持有的 Nintendo Switch 正版游戏中自行导出。

## 安装与首次启动

需要 Android 11 或以上、ARM64 处理器和 Vulkan 驱动。安装包见[最新发行版](https://github.com/yun-wulian/mhrise-recomp/releases/latest)。

1. 从自己持有的 Switch 游戏提取 **16.0.1** 的 ExeFS 和已合并同一更新的完整 RomFS，将两者放进平板上的同一个游戏文件夹。ExeFS 包含 `main` 和 `main.npdm`，RomFS 包含 `re_chunk_000.pak` 和更新文件 `re_chunk_000.pak.patch_015.pak`。如有 DLC，保留各 DLC 文件夹名称中的标题 ID。
2. 打开 **MHRise**。首次授权页右上角有 **⚙ Language / 语言**，点开选择 **简体中文**。首次设置和游戏内的齿轮菜单共用这个选择，游戏文本语言由游戏自身管理。
3. 点 **打开权限设置**，允许访问所有文件，返回后打开上面的游戏文件夹，点 **使用此文件夹**。启动器会检查游戏版本和文件是否齐全。
4. 再选择一个可写的**数据文件夹**。存档、MOD、着色器缓存和 `log.txt` 都放在这里，便于备份。如果在同一设备并排测试两个安装包，可以共用游戏文件夹，并为每个包选择各自的数据文件夹。
5. 点 **启动游戏**。此后启动器会记住路径。首次进入新区域时，着色器和管线编译可能带来短暂停顿。

## 操作、MOD 和问题反馈

- **操作**　连接实体手柄后，虚拟按键和摇杆自动隐藏，齿轮菜单和触屏拖动仍可用。断开最后一只手柄后，按原设置恢复显示。游戏内齿轮可以调整按键布局、画面分辨率、帧率上限和 GPU 驱动，也可以在「显示 → 菜单语言」切换中英文。
- **MOD**　在「MOD 与金手指」中导入或启停 MOD。Lua 运行时启动后可直接重载纯脚本 MOD。资源包以及同时包含资源和脚本的 MOD 在重启游戏后生效。「MOD 界面」可查看脚本控件及运行错误。
- **存档**　齿轮菜单中的「存档」提供导入、导出和备份恢复。升级日常版时使用相同包名和签名覆盖安装，即可继续使用原数据目录。
- **渲染问题**　在问题出现的位置打开「调试 → 问题报告」，点「捕获当前帧」，稍等几秒后点「保存报告」。反馈时附上设备型号、GPU 驱动、复现动作和报告中的相关日志。

## 从源码构建

Linux 或 macOS 构建需要 Git、Bash、curl、zip、Python 3、JDK 17+、CMake 3.22+，以及 Android SDK 中的 NDK `28.2.13676358`、Build Tools `35.0.0` 和平台 `android-33`。先将 `ANDROID_HOME` 指向本机 SDK，然后在仓库根目录运行：

```bash
bash tools/setup.sh
bash tools/build_ffmpeg.sh
LUA_LAB=1 bash tools/build_apk.sh
```

产物为 `build/mhriseport.apk`，应用名为 **MHRise**，包名为 `io.github.iamwilmar.mhriseport.lab`。默认使用本机 Android 调试密钥。FFmpeg 提供部分过场视频需要的 MP4 解封装后备路径。构建 APK 无需游戏文件，运行时再选择自己的文件夹。

Windows 上准备好第三方源码后，可运行以下 PowerShell 命令，从源码构建发行包。SDK 需安装 Build Tools `36.0.0`、平台 `android-34` 和 CMake `3.22.1`。路径按本机配置填写。

```powershell
python tools/build_lua_lab.py --source-release --sdk $env:ANDROID_HOME --ndk $env:ANDROID_NDK_HOME --jdk $env:JAVA_HOME
```

已签名合并包的增量更新使用 `--baseline-apk`，其中 `--setup-only` 更新首次设置和菜单，`--settings-update` 同时构建渲染库。签名与输出路径参数见[打包脚本](tools/build_lua_lab.py)。

## 源码导航

| 路径 | 内容 |
| --- | --- |
| [android/java](android/java) | 首次设置、菜单、手柄和 MOD 管理 |
| [runtime/src/loader](runtime/src/loader) | ARM64 加载器、Lua 运行时接入和 MOD UI |
| [runtime/src/nnsdk/nvn](runtime/src/nnsdk/nvn) | NVN 命令、显存、资源视图、Vulkan 管线和呈现 |
| [runtime/recompiler](runtime/recompiler) | Maxwell 着色器翻译、SPIR-V 生成与优化 |
| [tools/tests](tools/tests) | 主机回归测试与 Android 离屏验证负载 |
| [开发文档](docs/README.md) · [设置项参考](docs/settings.md) | 架构、GPU 差异、调试方法和配置开关 |

## 来源与许可

原始 Android 移植来自 IamWilmar，本分支基于 klark231 的后续开发。yun-wulian 分支截至 2026-11-22 的修改包括 Lua/MOD 集成、NVN/Vulkan 资源管理与粒子渲染修正，以及 Android 首次设置和中文菜单。[原始项目](https://github.com/IamWilmar/mhrise-recomp) · [直接上游](https://github.com/klark231/mhrise-recomp)

本项目的移植代码沿用 **GNU General Public License v3.0 或更高版本**（`GPL-3.0-or-later`）。修改和再分发时请保留版权与许可证声明，分发 APK 时同时提供该构建对应的完整源码和构建脚本。第三方组件按各自许可证使用。[许可证全文](LICENSE) · [第三方声明](THIRD_PARTY_NOTICES.md)
