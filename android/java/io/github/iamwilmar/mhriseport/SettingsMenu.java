package io.github.iamwilmar.mhriseport;

import io.github.iamwilmar.mhriseport.SetupActivity.MenuLanguage;

import android.app.Dialog;
import android.content.Intent;
import android.content.res.ColorStateList;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.ColorDrawable;
import android.graphics.drawable.GradientDrawable;
import android.util.DisplayMetrics;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.InputDevice;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.SeekBar;
import android.widget.TextView;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.IntConsumer;
import java.util.function.IntSupplier;


// Settings menu drawn over the game, styled after the game's own menus: dark wine panels with a bronze border, gold
// titles over the red and green rule, and a "?" on each row that expands its help text. Opened from the gear button
// of the touch controls. Sections on the left, the selected section's rows on the right.
// Control settings are saved and applied right away (GameActivity.refresh). Display, Graphics, Performance and Debug
// are the game's switches: empty files in the game folder (runtime/src/nnsdk/nvn/flags.h) that the game checks once a
// second. The ones only read at start-up say so in their help.
final class SettingsMenu {
    static final int PANEL = 0xF0201010, ROW = 0xFF2C1614, EDGE = 0xFFC49646, GOLD = 0xFFE2B65C, CREAM = 0xFFF0E4C8,
            DIM = 0xFFB09C84, RED = 0xFFB0221E, GREEN = 0xFF508E36;

    private static int section = 0;  // last one opened (for the life of the process)

    private final GameActivity activity;
    private final Controls cfg;
    private final Dialog dialog;
    private final LinearLayout list, sections;
    private final ScrollView scroll;
    private final TextView sub;
    private final File folder;  // the data folder (files/carpeta.txt), where mhr.ini lives
    private final Ini ini;
    private final float scale;  // menu size (Display > Menu size): text and spacing, 0.8 by default
    static final String PREFS = "menu", SCALE_KEY = "escala";
    static final int SCALE_DEFAULT = 80;

    private final boolean chinese;
    private final String[] mapNames;

    private String tr(String english, String simplifiedChinese) {
        return chinese ? simplifiedChinese : english;
    }

    SettingsMenu(GameActivity activity, Controls cfg) {
        this.activity = activity;
        this.cfg = cfg;
        chinese = MenuLanguage.usesChinese(activity);
        mapNames = mappingNames();
        this.scale = sizeFactor(activity);
        dialog = new Dialog(activity);
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE);

        LinearLayout root = new LinearLayout(activity);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackground(plate(PANEL, 2, 0));

        // top bar: "SETTINGS ◆ <section>" and the X, with the red and green rule under it
        LinearLayout top = new LinearLayout(activity);
        top.setGravity(Gravity.CENTER_VERTICAL);
        top.setPadding(dp(18), dp(10), dp(10), dp(10));
        TextView title = text(tr("SETTINGS", "设置"), 22, GOLD, true);
        title.setTypeface(Typeface.create(Typeface.SERIF, Typeface.BOLD));
        sub = text("", 17, CREAM, false);
        TextView close = text("✕", 22, CREAM, false);
        close.setPadding(dp(14), dp(4), dp(14), dp(4));
        close.setBackground(plate(0xFF1A0C09, 1, 2));
        close.setOnClickListener(v -> dialog.dismiss());
        top.addView(title);
        top.addView(sub, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        top.addView(close);
        root.addView(top);
        root.addView(rule(RED, 2));
        root.addView(rule(GREEN, 1));

        folder = gameFolder(activity);
        ini = new Ini(folder);
        gpu = GpuInfo.read(folder);
        LinearLayout body = new LinearLayout(activity);
        sections = new LinearLayout(activity);
        sections.setOrientation(LinearLayout.VERTICAL);
        sections.setPadding(dp(10), dp(8), dp(6), dp(10));
        ScrollView side = new ScrollView(activity);
        side.addView(sections);
        body.addView(side, new LinearLayout.LayoutParams(dp(176), ViewGroup.LayoutParams.MATCH_PARENT));
        View divider = new View(activity);
        divider.setBackgroundColor(EDGE);
        body.addView(divider, new LinearLayout.LayoutParams(dp(1), ViewGroup.LayoutParams.MATCH_PARENT));
        list = new LinearLayout(activity);
        list.setOrientation(LinearLayout.VERTICAL);
        list.setPadding(dp(12), dp(6), dp(12), dp(12));
        scroll = new ScrollView(activity);
        scroll.addView(list);
        body.addView(scroll, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1));
        root.addView(body, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));
        build();

        dialog.setContentView(root);
        Window w = dialog.getWindow();
        if (w != null) {
            w.setBackgroundDrawable(new ColorDrawable(Color.TRANSPARENT));
            DisplayMetrics m = activity.getResources().getDisplayMetrics();
            w.setLayout(Math.min(dp(940), (int) (m.widthPixels * 0.94f)), (int) (m.heightPixels * 0.92f));
        }
    }

    void show() { dialog.show(); }

    private void changed() { activity.refresh(); }

    // ---- sections -------------------------------------------------------------------------------------------------------

    private String[] sectionNames() {
        return new String[] {tr("Display", "显示"), tr("Graphics", "图形"), tr("GPU driver", "GPU 驱动"), tr("Frame generation", "帧生成"), tr("Performance", "性能"),
                tr("Touch controls", "触控"), tr("Combos", "组合键"), tr("Controller", "实体手柄"),
                tr("Multiplayer", "多人游戏"), tr("Saves", "存档"), tr("Mods & cheats", "MOD 与金手指"), tr("Debug", "调试")};
    }

    private void build() {
        String[] names = sectionNames();
        sections.removeAllViews();
        for (int i = 0; i < names.length; i++) {
            final int k = i;
            boolean on = k == section;
            TextView b = text(names[k], 15, on ? GOLD : CREAM, on);
            b.setPadding(dp(12), dp(11), dp(8), dp(11));
            b.setBackground(plate(on ? 0xFF3E2616 : ROW, on ? 2 : 1, 2));
            b.setOnClickListener(v -> {
                section = k;
                build();
                scroll.scrollTo(0, 0);
            });
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT);
            lp.topMargin = dp(4);
            sections.addView(b, lp);
        }
        sub.setText("  ◆  " + names[section]);
        list.removeAllViews();
        switch (section) {
            case 0: buildDisplay(); break;
            case 1: buildGraphics(); break;
            case 2: buildDriver(); break;
            case 3: buildFrameGen(); break;
            case 4: buildPerformance(); break;
            case 5: buildTouch(); break;
            case 6: buildCombos(); break;
            case 7: buildController(); break;
            case 8: buildMultiplayer(); break;
            case 9: buildSaves(); break;
            case 10: buildMods(); break;
            default: buildDebug(); break;
        }
    }

    private void buildDisplay() {
        header(tr("Menu", "菜单"));
        String language = tr("Menu language", "菜单语言");
        int selected = chinese ? 1 : 0;
        row(language, MenuLanguage.NAMES[selected] + "  ▾", tr("Language for setup and settings.", "首次设置和齿轮菜单使用的语言。"),
                step -> dropDown(language, MenuLanguage.NAMES, selected, i -> {
                    MenuLanguage.setChinese(activity, i == 1);
                    dialog.dismiss();
                    new SettingsMenu(activity, cfg).show();
                }), false);
        number(tr("Menu size", "菜单大小"), tr("The size of this menu and the MOD UI, on top of what fits your screen.", "本菜单和 MOD UI 的大小（在适配屏幕的基础上调整）。"), 60, 130, 5, " %", () -> menuScale(activity), v -> {
                    try {
                        activity.getSharedPreferences(PREFS, android.content.Context.MODE_PRIVATE).edit().putInt(SCALE_KEY, v).apply();
                    } catch (RuntimeException ignored) {
                        // (no preferences: keep the current size)
                    }
                    dialog.dismiss();
                    new SettingsMenu(activity, cfg).show();  // (sizes are fixed at build time, so reopen with the new one)
                });
        action(resetArmed ? tr("Tap again to restore the defaults", "再点一次恢复默认设置") : tr("Restore default settings", "恢复默认设置"),
                tr("Puts every setting of this menu back as it comes: graphics, performance, debug and the driver "
                        + "variables. Your touch layout, combos and button mapping stay. Some settings need a restart.", "重置图形、性能、调试及驱动变量。保留触控布局、组合键和按键映射。部分设置重启后生效。"),
                () -> {
                    if (!resetArmed) {
                        resetArmed = true;
                        build();
                        return;
                    }
                    resetArmed = false;
                    ini.reset(SWITCHES);
                    build();
                });
        header(tr("Resolution", "分辨率"));
        note(tr("The game's 3D resolution. On the handheld console it's 960x540 inside a 720p picture, which is then scaled "
                        + "to the screen. Higher: sharper, and more work for the GPU.", "掌机模式原本以 960×540 绘制 3D 场景，再将 720p 输出缩放至屏幕。提高分辨率会让画面更清晰，也会增加 GPU 负载。"));
        exclusive(tr("Base resolution", "基础分辨率"),
                tr("540p is the game's own. Lower: blurrier, much less GPU work, for weak phones. 720p draws the whole scene "
                        + "at 1280x720, with no scaling inside the game. In docked mode the same steps apply to the "
                        + "game's docked resolution. Applies at once.", "游戏默认 540p。调低可减少 GPU 负载；720p 以 1280×720 直接绘制场景。底座模式按其基础分辨率同比例调整。立即生效。"),
                new String[] {"540p (960x540)", "240p (427x240)", "360p (640x360)", "432p (768x432)", "480p (854x480)", "576p (1024x576)",
                        "648p (1152x648)", "720p (1280x720)"},
                new String[] {null, "resolution_240", "resolution_360", "resolution_432", "resolution_480", "resolution_576",
                        "resolution_648", "resolution_720"});
        exclusive(tr("Aspect ratio", "宽高比"),
                tr("The shape of the picture on your screen, like Eden's setting. 16:9 is the game's own, with black bars at "
                        + "the sides on a long phone. The others stretch the game's picture to that shape; Stretch fills the "
                        + "whole screen. Applies at once.", "默认 16:9，宽屏两侧留黑。可拉伸至指定比例或铺满屏幕。立即生效。"),
                new String[] {tr("Default (16:9)", "默认（16:9）"), tr("Force 4:3", "强制 4:3"), tr("Force 21:9", "强制 21:9"), tr("Force 16:10", "强制 16:10"), tr("Stretch to screen", "铺满屏幕")},
                new String[] {null, "aspect_4_3", "aspect_21_9", "aspect_16_10", "aspect_stretch"});
        flag(tr("Expand into the cutout", "使用摄像头挖孔区域"),
                tr("Use the strip beside the camera cutout too, as in Eden and Azahar. Off: Android keeps the picture clear "
                        + "of it. With a wider aspect ratio or Stretch, On shows the picture right to the edge. Takes "
                        + "effect on the next start.", "允许游戏延伸到摄像头挖孔旁的区域，配合宽屏比例或铺满屏幕可显示到边缘。下次启动生效。"),
                "cutout", false);
        flag(tr("Show Android's bars", "显示 Android 系统栏"),
                tr("Keep the status and navigation bars on screen while playing. Off: they're hidden, and a swipe in from "
                        + "the edge brings them back for a moment. Takes effect the next time you come back to the game.", "显示状态栏和导航栏。关闭后仍可从边缘滑动临时显示。下次返回游戏时生效。"),
                "system_bars", false);
        toggle(tr("Draw controls with the GPU", "使用 GPU 绘制触控按键"),
                tr("The touch controls drawn by the GPU, as Android usually does. Off: drawn by the CPU, which keeps them "
                        + "away from the phone's OpenGL driver; Mali's crashed there while hunting. Off by default on MediaTek "
                        + "and Exynos. Takes effect the next time you start the game.", "开启后使用 Android GPU 绘制触控界面，关闭后使用 CPU。部分 Mali 设备的 OpenGL 驱动可能在狩猎时崩溃，联发科和 Exynos 设备默认关闭。下次启动生效。"),
                () -> GameActivity.touchOnGpu(ini) ? 1 : 0, v -> ini.set("touch_gpu", v != 0 ? "1" : "0"));
        flag(tr("Docked mode", "底座模式"),
                tr("The game thinks it's on the dock: its TV picture (1080p output, about twice the pixels of handheld) and "
                        + "your controls as player 1's Pro Controller. Much more work for the GPU: for strong phones. Takes "
                        + "effect on the next start.", "使用主机的电视模式（1080p 输出，像素约为掌机的两倍），控制器作为玩家一的 Pro 手柄。GPU 负载较高，下次启动生效。"),
                "dock_mode", false);
        flag(tr("Dynamic resolution", "动态分辨率"),
                tr("If the GPU can't finish a frame in time, it lowers the resolution in steps (down to 73 % of the base) "
                        + "and raises it again when there's room. Off: always the base.", "GPU 来不及完成一帧时逐级降低分辨率，最低为基础分辨率的 73%；负载降低后再升高。关闭后固定为基础分辨率。"),
                "no_dynamic_resolution", true);
        flag(tr("FSR upscaling", "FSR 超分辨率"),
                tr("AMD FSR 1 takes the 720p picture to the screen sharper than a plain scale. Off: linear scaling, a "
                        + "little cheaper.", "使用 AMD FSR 1 将 720p 画面放大至屏幕。关闭后采用开销较低的线性缩放。"),
                "no_fsr", true);

        header(tr("Picture", "图像"));
        final String[] caps = {null, "fps_20", "fps_25", "fps_40", "fps_45", "60fps"};
        choice(tr("FPS limit", "帧率上限"),
                tr("The game measures how long each frame takes, so between 20 and 60 fps it runs at its normal speed: "
                        + "fewer fps save GPU, heat and battery with no slow motion. 30 is the game's own (loading screens "
                        + "always ask for 60). 40 is even on 120 Hz screens; on a 60 Hz one, 40 and 45 judder a little. "
                        + "Applies at once.", "20 至 60 帧之间游戏保持正常速度。限制帧率可降低功耗与发热。游戏默认 30 帧，加载画面请求 60 帧。40 帧适合 120 Hz 屏，在 60 Hz 屏上 40 和 45 帧可能轻微抖动。立即生效。"),
                new String[] {tr("30 (the game's)", "30（游戏默认）"), "20", "25", "40", "45", "60"}, () -> {
                    for (int i = caps.length - 1; i > 0; i--)
                        if (has(caps[i])) return i;
                    return 0;
                }, v -> {
                    for (int i = 1; i < caps.length; i++) put(caps[i], i == v);
                });
        flag(tr("FPS counter", "帧率计数器"), tr("The number in the corner.", "在屏幕角落显示帧率。"), "no_fps",
                true);

        header(tr("Performance overlay", "性能监视"));
        note(tr("More lines under the FPS counter, each on its own. With any of them on, the counter becomes a small box "
                + "that also shows how many of the fps are real when frame generation is on. All apply at once.", "在帧率下方显示所选指标。启用帧生成时也会显示实际渲染帧率。立即生效。"));
        flag(tr("Frame time", "帧时间"), tr("How long the game's frames take (ms), and the 1 % low: the fps of its slowest frames, "
                + "which is what you feel as stutter.", "显示每帧耗时（毫秒）及 1% Low 帧率，反映最慢一组帧的表现。"), "no_hud_frametime", true);
        flag(tr("CPU usage", "CPU 占用率"), tr("How much of the phone's CPU the game uses, all cores together.", "游戏使用的 CPU 比例，合计所有核心。"), "no_hud_cpu", true);
        flag(tr("GPU usage", "GPU 占用率"), tr("How busy the GPU is. Only where the phone lets apps read it (many Adreno phones do; most "
                + "others show N/A).", "GPU 的繁忙程度。部分 Adreno 设备允许读取；无法读取时显示 N/A。"), "no_hud_gpu", true);
        flag(tr("Memory", "内存"), tr("The game's own RAM, then the phone's used / total.", "显示游戏内存，以及设备已用和总内存。"), "no_hud_ram", true);
        flag(tr("Heat", "发热"), tr("How close the phone is to slowing itself down from heat (Android's own measure, Android 12 and "
                + "later): at 100 % it throttles hard. Yellow from 80 %, red from 95 %.", "Android 12 及以上显示系统热压力。80% 起为黄色，95% 起为红色，达到 100% 时会明显降频。"), "no_hud_heat", true);
        flag(tr("Shaders building", "正在编译着色器"), tr("Pipelines being made in the background right now. Above 0, a new effect or area is "
                + "being prepared: short hitches then are that.", "正在后台构建的管线数量。大于零时正在准备新区域或特效，可能出现短暂停顿。"), "no_hud_shaders", true);
    }

    private void buildGraphics() {
        header(tr("Effects", "特效"));
        flag(tr("Shadows", "阴影"),
                tr("Those of the hunters, monsters and scenery. Off, the shadow maps aren't drawn: less work for a weak "
                        + "GPU. Applies at once.", "绘制猎人、怪物及场景的阴影。关闭可减少 GPU 负载。立即生效。"),
                "no_shadows", true);
        note(gameStatus());
        exclusive(tr("Shadow quality", "阴影质量"),
                tr("The size of the shadow maps, as a share of the game's own. Lower: blurrier shadow edges and a lot less "
                        + "GPU work in one of the heaviest passes. Applies at once.", "相对于游戏默认值的阴影贴图大小。调低后阴影边缘更模糊，阴影绘制的 GPU 开销也会降低。立即生效。"),
                new String[] {tr("Game's", "游戏默认"), tr("Half", "二分之一"), tr("Quarter", "四分之一")}, new String[] {null, "shadow_half", "shadow_quarter"});
        flag(tr("Dynamic shadows", "动态阴影"),
                tr("Shadows of things that move (hunters, monsters, swaying plants), like the PC's option. Off: only the "
                        + "scenery's shadows, a lot less shadow work. Applies at once.", "绘制猎人、怪物和摇曳植物等移动物体的阴影。关闭后仅保留场景阴影。立即生效。"),
                "no_dynamic_shadows", true);
        flag(tr("Foliage sway", "植被摆动"),
                tr("Grass and leaves moving in the wind, like the PC's option. Off: they stand still, a bit less vertex "
                        + "work. Applies at once.", "草木随风摆动。关闭可减少顶点处理。立即生效。"),
                "no_foliage_sway", true);
        exclusive(tr("Effects resolution (experimental)", "特效分辨率（实验性）"),
                tr("The size of the buffer the game draws particles, smoke and other see-through effects into, as steps "
                        + "below the game's own (which is already reduced). Lower: blurrier effects, less GPU work in "
                        + "big fights. Applies at once.", "粒子、烟雾等半透明特效的绘制分辨率。游戏默认已降低，继续调低会使特效变模糊，同时减轻大型战斗的 GPU 负载。立即生效。"),
                new String[] {tr("Game's", "游戏默认"), tr("One step lower", "降低一级"), tr("Two steps lower", "降低两级")},
                new String[] {null, "effects_lower", "effects_lowest"});
        flag(tr("Ambient occlusion (experimental)", "环境光遮蔽（实验性）"),
                tr("The soft darkening where things meet (the game's geometry AO). Off: flatter look, less GPU work. "
                        + "Applies at once.", "在物体接触处添加柔和的暗部。关闭后画面更平，GPU 负载更低。立即生效。"),
                "no_ao", true);
        flag(tr("Fast depth test", "快速深度测试"),
                tr("The colour pass skips hidden pixels earlier (it uses the depth test the GPU can reject fast with), "
                        + "which can give a few more FPS, mostly on Adreno. Grass and leaves keep the game's own test so "
                        + "they don't turn into solid cards. Applies at once.", "在颜色绘制时提前剔除被遮挡的像素，部分 Adreno 设备可提升帧率。草木保留游戏的深度测试。立即生效。"),
                "fast_depth", false);
        header(tr("Detail", "细节"));
        flag(tr("Occlusion results", "遮挡查询结果"),
                tr("Lets the game see which small things its GPU test found visible, so endemic life, wirebugs and some "
                        + "lights get drawn. Off: they stay invisible (you can still gather them). Costs almost nothing. "
                        + "Applies at once.", "将 GPU 可见性结果提供给游戏，用于绘制环境生物、翔虫和部分光源。关闭后仍能采集不可见的对象。开销很小，立即生效。"),
                "no_readback_copies", true);
        if (gpu.noBc()) {
            flag(tr("Textures as ASTC (Mali)", "使用 ASTC 纹理（Mali）"),
                    tr("On GPUs that can't read the game's BC textures (Mali), keeps them compressed as ASTC instead of 4-8 "
                            + "times bigger: less memory traffic every frame. The first time a texture loads it's encoded on the "
                            + "CPU (slower loading and some stutter), then it comes from a cache in the app's storage. Restart the "
                            + "game to apply.", "在无法读取 BC 纹理的 GPU 上转码为 ASTC，可节省内存和带宽。首次载入需 CPU 编码，可能较慢；之后使用应用缓存。重启游戏生效。"),
                    "bc_astc", false);
            action(tr("Clear the ASTC cache", "清除 ASTC 缓存"),
                    tr("Deletes the encoded textures (", "删除应用内已编码的纹理（") + astcCacheSize() + tr(") from the app's storage. They're encoded again as "
                            + "they load. Uninstalling the app deletes them too.", "）。下次加载时重新编码；卸载应用也会删除它们。"),
                    () -> {
                        clearAstcCache();
                        build();
                    });
        }
        exclusive(tr("Model detail", "模型细节"),
                tr("How soon models switch to their simpler versions with distance (the PC version's mesh quality). Lower: "
                        + "less work for the GPU, simpler models a little closer. Applies at once.", "调整随距离切换低细节模型的时机。调低后，更近的模型也会简化，GPU 负载随之下降。立即生效。"),
                new String[] {tr("Game's", "游戏默认"), tr("Low", "低"), tr("Lowest", "更低"), tr("Minimum", "最低"), tr("High", "高")},
                new String[] {null, "model_detail_low", "model_detail_lowest", "model_detail_min", "model_detail_high"});
        exclusive(tr("Texture streaming", "纹理流送"),
                tr("How many of the largest texture sizes the game skips loading. Fewer: blurrier textures up close, less "
                        + "memory and bandwidth. Applies as textures load.", "跳过最高的若干级纹理。较小的纹理在近处更模糊，但可节省内存和带宽。加载纹理时生效。"),
                new String[] {tr("Game's", "游戏默认"), tr("One size smaller", "缩小一级"), tr("Two sizes smaller", "缩小两级")},
                new String[] {null, "texture_bias_1", "texture_bias_2"});
        header(tr("Textures", "纹理"));
        {
            final String[] files = {null, "bindless_arrays", "no_bindless_arrays"};
            choice(tr("Texture arrays (bindless)", "纹理数组（无绑定）"),
                    tr("How shaders read many of the game's textures. Auto: off with Qualcomm's driver on an Adreno 7xx "
                            + "or 8xx, which can't compile many of those shaders (things come out black); on elsewhere. "
                            + "Turn them off if your driver draws things black. Takes effect on the next start.", "控制着色器读取多张纹理的方式。自动模式在采用高通系统驱动的 Adreno 7xx 和 8xx 上关闭，在其他设备上开启。若纹理全黑可尝试关闭。下次启动生效。"),
                    new String[] {tr("Auto", "自动"), tr("On", "开"), tr("Off", "关")}, () -> {
                        for (int i = files.length - 1; i > 0; i--)
                            if (has(files[i])) return i;
                        return 0;
                    }, v -> {
                        for (int i = 1; i < files.length; i++) put(files[i], i == v);
                    });
        }
        flag(tr("Anisotropic filtering", "各向异性过滤"),
                tr("Sharper floors and walls at an angle in the distance. Off: a little cheaper for a weak GPU.", "让远处倾斜的地面与墙面更清晰。关闭可略微降低 GPU 负载。"),
                "no_af", true);
        if (gpu.eds || gpu.vids) {  // (only where the driver has them; known after the game's first start)
            header(tr("Extensions", "扩展"));
            if (gpu.eds)
                flag(tr("Extended dynamic state", "扩展动态状态"),
                        tr("Cull mode, depth test and stencil set per draw instead of built into each pipeline: far fewer "
                                + "pipelines to build, so fewer hitches when new effects or areas show up. Off by default: "
                                + "some drivers have bugs with it. Takes effect on the next start.", "每次绘制时设置剔除、深度与模板状态，可减少管线数量及初见特效时的卡顿。部分驱动存在兼容问题，默认关闭。下次启动生效。"),
                        "dynamic_state", false);
            if (gpu.vids)
                flag(tr("Vertex input dynamic state", "动态顶点输入"),
                        tr("The vertex layout set per draw too: fewer pipelines again (a mesh with another layout shares "
                                + "one). Off by default, for the same reason. Takes effect on the next start.", "每次绘制时设置顶点布局，便于不同网格共享管线。部分驱动存在兼容问题，默认关闭。下次启动生效。"),
                        "vertex_input_state", false);
        }
    }

    // ---- GPU driver and frame generation (Drivers: the same files as the first-start screen, DriverActivity) ----------

    private static String sessionDriver;  // the driver the game started with (it can't change until the next start)
    private static boolean sessionDriverRead;

    private void buildDriver() {
        if (!sessionDriverRead) {
            sessionDriverRead = true;
            sessionDriver = Drivers.chosen(activity);
        }
        String failed = Drivers.takeFailure(activity);
        if (failed != null) note(tr("The last driver didn't work (", "上次选择的驱动加载失败（") + failed + tr("): the system's is in use.", "），当前使用系统驱动。"));
        header(tr("Driver", "驱动"));
        note(tr("The phone's own driver, or one you add from a .zip (meta.json and the .so, as for AdrenoTools: Turnip and "
                + "the like). Adreno (Snapdragon) only, at your own risk. A change takes effect the next time the game "
                + "starts; if the game can't draw with the new one, it goes back to the system's by itself.", "使用系统驱动，或从包含 meta.json 和 .so 文件的 AdrenoTools 驱动压缩包导入 Turnip 等驱动。自定义驱动仅支持 Adreno。下次启动生效；若无法绘制，将恢复系统驱动。"));
        List<Drivers.Driver> drivers = Drivers.list(activity);
        String chosen = Drivers.chosen(activity);
        String[] names = new String[drivers.size() + 1];
        names[0] = tr("System driver", "系统驱动");
        int current = 0;
        for (int i = 0; i < drivers.size(); i++) {
            Drivers.Driver d = drivers.get(i);
            names[i + 1] = d.name + (d.detail.isEmpty() ? "" : " (" + d.detail + ")");
            if (d.dir.getName().equals(chosen)) current = i + 1;
        }
        pick(tr("Driver in use", "使用的驱动"), tr("The one the game uses from its next start.", "下次启动游戏时加载此驱动。"), names, current, v -> {
            try {
                Drivers.choose(activity, v == 0 ? null : drivers.get(v - 1).dir);
            } catch (IOException e) {
                android.widget.Toast.makeText(activity, tr("Couldn't save it: ", "保存失败：") + e.getMessage(), android.widget.Toast.LENGTH_LONG)
                        .show();
            }
        });
        action(tr("Add a driver (.zip)", "添加驱动（.zip）"), tr("Pick a driver's zip. It's added to the list; choose it above to use it.", "选择驱动压缩包，添加后可在上方选用。"), () -> {
            dialog.dismiss();
            activity.importDriver();
        });
        if (!drivers.isEmpty()) {
            String[] own = Arrays.copyOfRange(names, 1, names.length);
            action(tr("Remove a driver", "移除驱动"), tr("Deletes an added driver. If it's the one in use, the system's is used instead.", "删除已添加的驱动。若正在使用该驱动，下次将使用系统驱动。"),
                    () -> dropDown(tr("Remove a driver", "移除驱动"), own, -1, i -> dropDown(tr("Delete ", "删除 ") + drivers.get(i).name + "?",
                            new String[] {tr("Keep it", "保留"), tr("Delete it", "删除")}, -1, k -> {
                                if (k == 1) {
                                    try {
                                        Drivers.remove(activity, drivers.get(i).dir);
                                    } catch (IOException ignored) {
                                        // (left as it was)
                                    }
                                }
                                build();
                            })));
        }
        boolean changed = chosen == null ? sessionDriver != null : !chosen.equals(sessionDriver);
        if (changed) {
            note(tr("The game keeps the driver it started with. Close it and open the app again to use the new one.", "本次游戏仍在使用启动时的驱动。重启应用可使用新选择的驱动。"));
            action(tr("Close the game", "关闭游戏"), tr("Closes the app now, so the next start uses the chosen driver.", "关闭应用，下次启动时使用所选驱动。"), () -> {
                activity.finishAffinity();
                android.os.Process.killProcess(android.os.Process.myPid());
            });
        }
    }

    private void buildFrameGen() {
        if (folder == null) {
            note(tr("Pick the game folder first.", "请先选择游戏目录。"));
            return;
        }
        File dll = Drivers.losslessDll(activity);
        File builtin = new File(folder, "fg_builtin");
        boolean own = !dll.isFile() || builtin.exists();
        header(tr("Frame generation", "帧生成"));
        note(tr("Shows about twice the game's fps (30 → 60) by making a picture between each two of the game's. With your own "
                + "Lossless.dll (from Lossless Scaling on Steam, Windows: in the program's folder; not included with the "
                + "app) it can use Lossless Scaling's; otherwise the port's own.", "在相邻两帧之间生成中间帧，例如将 30 帧提升至 60 帧。可用内置算法，也可从自己的 Windows 版 Lossless Scaling 程序目录导入 Lossless.dll。"));
        toggle(tr("Frame generation", "帧生成"), tr("On or off while you play (the same as the in-game panel's row).", "游戏中可随时开关，与游戏内面板的选项相同。"),
                () -> has("lsfg") ? 1 : 0, v -> ini.set("lsfg", v != 0 ? "1" : "0"));
        if (dll.isFile()) {
            String[] kinds = {"Lossless Scaling", tr("Built-in", "内置")};
            String size = String.format(java.util.Locale.ROOT, "%.1f MB", dll.length() / 1048576.0);
            pick(tr("Generator", "帧生成算法"), tr("Lossless Scaling: sharper, heavier on the GPU (Lossless.dll, ", "Lossless Scaling 画质更清晰，GPU 开销也更大（Lossless.dll，") + size + tr("). Built-in: lighter, "
                    + "runs on any GPU. Takes effect the next time the game starts.", "）。内置算法较轻量，适用于各种 GPU。下次启动生效。"), kinds, own ? 1 : 0, k -> {
                if (k == 0) Drivers.deleteTree(builtin);
                else builtin.mkdirs();
            });
        }
        exclusive(tr("Target", "目标帧率"),
                tr("Off: one generated picture between each two real ones (30 → 60). With a target, as many as it takes to "
                        + "reach it, like Lossless Scaling's adaptive mode: at 20 real fps two (60), on a 120 Hz screen up "
                        + "to three. Never above the screen's refresh rate, and each extra picture costs about as much GPU "
                        + "as the first, so on a busy GPU it can slow the real frames down. The built-in generator does "
                        + "this more cheaply than Lossless Scaling's. Applies at once.", "关闭时每两帧间插入一帧。设定目标后按需插帧，例如从 20 帧到 60 帧需插入两帧，最高不超过屏幕刷新率。每多插一帧都会增加 GPU 负载，可能拉低实际帧率。内置算法开销较低。立即生效。"),
                new String[] {tr("Off (2x)", "关闭（2 倍）"), "60 fps", "90 fps", "120 fps"},
                new String[] {null, "fg_target_60", "fg_target_90", "fg_target_120"});
        File half = new File(folder, "lsfg_flow_50"), full = new File(folder, "lsfg_flow_100");
        int level = full.exists() ? 2 : half.exists() ? 1 : 0;
        String[] presets = own ? new String[] {tr("Light (motion at 1/4)", "轻量（运动分辨率 1/4）"), tr("Balanced (motion at 1/3)", "均衡（运动分辨率 1/3）"), tr("Sharp (motion at 1/2)", "清晰（运动分辨率 1/2）")}
                               : new String[] {tr("Light (flow 25 %)", "轻量（光流 25%）"), tr("Balanced (flow 50 %)", "均衡（光流 50%）"), tr("Sharp (flow 100 %)", "清晰（光流 100%）")};
        pick(tr("Quality Preset", "质量预设"), tr("How finely ", "调整") + (own ? tr("the generator", "帧生成算法") : "Lossless Scaling") + tr(" follows motion. Higher is "
                + "sharper with less smearing, and heavier on the GPU (each step about twice the work). Applies at once.", "的运动追踪精度。越高越清晰、拖影越少，每升一级 GPU 工作量约翻倍。立即生效。"),
                presets, level, k -> {
                    Drivers.deleteTree(half);
                    Drivers.deleteTree(full);
                    if (k == 1) half.mkdirs();
                    if (k == 2) full.mkdirs();
                });
        action(dll.isFile() ? tr("Replace Lossless.dll", "替换 Lossless.dll") : tr("Import Lossless.dll", "导入 Lossless.dll"),
                tr("Pick Lossless.dll from your copy of Lossless Scaling. Takes effect the next time the game starts.", "从自己的 Lossless Scaling 程序目录中选择 Lossless.dll。下次启动生效。"), () -> {
                    dialog.dismiss();
                    activity.importLosslessDll();
                });
        if (dll.isFile())
            action(tr("Remove Lossless.dll", "移除 Lossless.dll"), tr("Deletes the imported copy; frame generation goes on with the built-in one.", "删除导入的副本，之后使用内置帧生成算法。"), () -> {
                dll.delete();
                build();
            });
    }

    // A row whose value opens a drop-down, whatever the number of options (choice() flips two on a tap)
    private void pick(String title, String help, String[] options, int current, IntConsumer set) {
        String shown = options[current];
        int paren = shown.indexOf(" (");
        row(title, (paren > 0 ? shown.substring(0, paren) : shown) + "  ▾", help, x -> dropDown(title, options, current, k -> {
            set.accept(k);
            changed();
            build();
        }), false);
    }

    private void buildPerformance() {
        header(tr("GPU work", "GPU 负载"));
        flag(tr("Skip empty GPU barriers", "跳过空 GPU 屏障"),
                tr("About half of the GPU's full stops in a frame come right after another one with nothing in between, and "
                        + "still make the GPU wait and flush. On: those are skipped (less waiting, a little less heat). Off "
                        + "by default: if graphics glitch with it on, turn it off. Applies at once.", "省略中间没有任务的连续 GPU 屏障，减少等待与发热。默认关闭，若开启后画面异常可关闭。立即生效。"),
                "tight_barriers", false);
        exclusive(tr("Pipeline worker threads", "管线编译线程数"),
                tr("Threads available to build GPU pipelines. More can finish shader warm-up sooner. "
                        + "A draw waits if its pipeline is still being built. Takes effect on the next start.", "可用于编译 GPU 管线的线程数。增加线程可加快预热；绘制仍会等待正在构建的管线。下次启动生效。"),
                new String[] {tr("2 (default)", "2（默认）"), "1", "3", "4"},
                new String[] {null, "pipeline_threads_1", "pipeline_threads_3", "pipeline_threads_4"});
        flag(tr("Asynchronous presentation", "异步呈现"),
                tr("Puts each finished frame on screen from a thread of its own (like Eden's setting), so the renderer starts "
                        + "the next frame instead of waiting for the screen. Can give a few fps back on a busy phone. Off by "
                        + "default: if the picture stutters or tears with it, turn it off. Applies at once.", "使用独立线程呈现已完成的帧，让渲染器提前开始下一帧。部分设备可提高帧率。若出现抖动或撕裂可关闭，默认关闭。立即生效。"),
                "async_present", false);
        flag(tr("GPU texture unswizzle", "GPU 纹理重排"),
                tr("Switch textures are stored in a tiled order, and each one is put back in plain rows before the GPU gets "
                        + "it. On: the GPU does that instead of the CPU, which then only copies each texture as it is, so "
                        + "loading an area or a monster stutters less (like Eden's option). Off by default until it's been "
                        + "tried on more phones. Applies at once.", "使用 GPU 将 Switch 的分块纹理重排为行布局，CPU 只需复制原始数据，可减轻区域或怪物加载时的卡顿。默认关闭，立即生效。"),
                "gpu_unswizzle", false);
        if (gpu.adreno()) {
            header("GPU");
            flag(tr("GPU boost (Adreno)", "GPU 加速（Adreno）"),
                    tr("Keeps the Adreno at its top clock while the game is open (libadrenotools' turbo, no root needed), "
                            + "so it doesn't slow down between frames. Hotter and more battery; the phone still throttles "
                            + "when it gets too hot. Turned off when you leave the game. Applies at once.", "通过 libadrenotools 在前台保持较高 GPU 频率，无需 root。会增加功耗与发热，过热时仍可能降频。离开游戏时关闭。立即生效。"),
                    "adreno_turbo", false);
        }
        header("CPU");
        flag(tr("Performance hints", "性能提示"),
                tr("Tells Android how long each frame should take and how much work it took, so it raises the clocks in "
                        + "time instead of letting them drop on every wait (Android 13 or newer). Usually a steadier pace.", "向 Android 13 及以上报告每帧的目标和实际耗时，帮助系统及时调频，使帧间隔更稳定。"),
                "no_adpf", true);
        flag(tr("Game threads on the fast cores", "将游戏线程绑定至大核"),
                tr("The three threads that build each frame go on the fast cores (if the phone has at least three). Off: "
                        + "Android places them.", "设备至少有三个高性能核心时，将三个主要帧处理线程绑定至大核。关闭后由 Android 调度。"),
                "no_affinity", true);
        header(tr("Memory and shaders", "内存与着色器"));
        flag(tr("Low memory", "低内存模式"),
                tr("Smaller caches, as on a phone with less than 5 GB (it's already like that there). Takes effect on the "
                        + "next start.", "减小缓存。内存不足 5 GB 的设备默认启用。下次启动生效。"),
                "low_memory", false);
        flag(tr("Shader warm-up", "着色器预热"),
                tr("At start it builds the pipelines seen in earlier runs, so they don't drop frames when they appear. "
                        + "Takes effect on the next start.", "启动时编译之前用过的管线，减少它们再次出现时的卡顿。下次启动生效。"),
                "no_warmup", true);
        flag(tr("Async shaders", "异步着色器"),
                tr("A shader the game hasn't used before is translated in the background, and what needs it shows a moment "
                        + "late the first time. Off: the game stops until it's translated (a short freeze).", "后台翻译首次出现的着色器，对应物体可能稍晚显示。关闭后游戏会等待翻译完成。"),
                "no_async_shaders", true);
        flag(tr("Pipelines on one thread", "单线程管线编译"),
                tr("No background threads or warm-up: more stutter, but some drivers crash building several at once "
                        + "(PowerVR is like that by itself). Takes effect on the next start.", "关闭后台编译及预热，可能增加卡顿，适用于并行编译会崩溃的驱动。PowerVR 默认启用。下次启动生效。"),
                "single_thread_pipelines", false);
    }

    // Local Play over the internet (runtime/src/nnsdk/ldn.cpp): hubs created in the game's Local Play are listed on the
    // hub servers, and the game's hub search shows theirs. Saved as plain text files in the game folder: hub_servers (one
    // per line, "#" in front for one switched off, "off" for none), hub_key, hub_hosts and hub_port. Read when a hub is
    // created or searched for, so changes apply without a restart. Not the MHGU hub servers: their owners asked for
    // MH Rise hubs to stay off them.
    private static final String[] DEFAULT_SERVERS = {"meoli.viewdns.net"};
    private static final String RETIRED_SERVER = "morerokk-mhgurecomp-ldnserver.duckdns.org";

    private List<String> lines(String name) {
        List<String> out = new ArrayList<>();
        if (folder == null) return out;
        try {
            for (String l : Files.readAllLines(new File(folder, name).toPath())) {
                l = l.trim();
                if (!l.isEmpty()) out.add(l);
            }
        } catch (IOException ignored) {
            // (not there yet)
        }
        return out;
    }

    private void writeLines(String name, List<String> list) {
        if (folder == null) return;
        try {
            Files.write(new File(folder, name).toPath(), (String.join("\n", list) + "\n").getBytes());
        } catch (IOException ignored) {
            // (storage refused: the row shows the old value)
        }
    }

    private List<String> servers() {
        List<String> l = lines("hub_servers");
        if (l.isEmpty()) return new ArrayList<>(Arrays.asList(DEFAULT_SERVERS));
        if (l.size() == 1 && l.get(0).equals("off")) return new ArrayList<>();
        l.removeIf(x -> x.replace("#", "").trim().startsWith(RETIRED_SERVER));  // (an older list saved with it)
        return l;
    }

    private void ask(String title, String guide, String initial, int max, java.util.function.Consumer<String> done) {
        new TextPrompt(activity, chinese).show(title, guide, initial, "", 0, max, false, false, text -> {
            if (text != null) done.accept(text.trim());
            build();
        });
    }

    private void buildMultiplayer() {
        note(tr("Local Play over the internet: hubs you create in the game's Local Play are listed on the hub servers below, "
                + "and searching for hubs shows the ones there. Only MH Rise players are listed (not MHGU's). No port "
                + "forwarding needed.", "通过互联网使用「本地联机」。创建或搜索集会区域时会使用下列服务器，只显示《崛起》的玩家，无需端口转发。"));
        header(tr("Matchmaking", "匹配"));
        List<String> key = lines("hub_key");
        row(tr("Matchmaking key", "匹配密钥"), key.isEmpty() ? tr("None", "无") : "••••",
                tr("Players who enter the same key only see each other's hubs. Empty: public, seen by everyone without a key. "
                        + "Only a hash of it is sent.", "使用相同密钥的玩家只能看到彼此的集会区域。留空参与公开匹配。仅发送密钥的哈希值。"),
                v -> ask(tr("Matchmaking key", "匹配密钥"), tr("Empty for public hubs", "公开匹配请留空"), key.isEmpty() ? "" : key.get(0), 64, text -> {
                    if (text.isEmpty()) new File(folder, "hub_key").delete();
                    else writeLines("hub_key", Arrays.asList(text));
                }), false);

        header(tr("Hub servers", "集会区域服务器"));
        List<String> list = servers();
        if (list.isEmpty()) note(tr("No hub servers: only the search hosts below.", "没有服务器，仅使用下方的直连主机。"));
        else note(tr("Tap a server to switch it on or off, or to remove it. Hubs are listed on, and searched for in, every "
                + "server that's on.", "点按服务器可启用、停用或移除。集会区域会在所有已启用的服务器上发布和搜索。"));
        for (int i = 0; i < list.size(); i++) {
            final int k = i;
            final boolean on = !list.get(i).startsWith("#");
            final String name = on ? list.get(i) : list.get(i).substring(1).trim();
            row(name, (on ? tr("On", "开") : tr("Off", "关")) + "  ▾", tr("On: hubs are listed and searched for here. Off: kept in the list, "
                    + "not used.", "开启后在此发布和搜索集会区域；关闭后保留地址。"), v -> dropDown(name, new String[] {tr("On", "开"), tr("Off", "关"), tr("Remove", "移除")}, on ? 0 : 1, pick -> {
                        List<String> l = servers();
                        if (k >= l.size()) return;
                        if (pick == 2) l.remove(k);
                        else l.set(k, pick == 0 ? name : "#" + name);
                        writeLines("hub_servers", l.isEmpty() ? Arrays.asList("off") : l);
                        build();
                    }), false);
        }
        action(tr("Add hub server", "添加集会区域服务器"), tr("A hub server's address: name or IP, and :port if it isn't 39510.", "输入服务器域名或 IP。端口不是 39510 时请附加端口号。"),
                () -> ask(tr("Add hub server", "添加集会区域服务器"), tr("host or host:port", "主机或主机:端口"), "", 128, text -> {
                    if (text.isEmpty() || text.startsWith("#")) return;
                    List<String> l = servers();
                    if (!l.contains(text) && !l.contains("#" + text)) l.add(text);
                    writeLines("hub_servers", l);
                }));
        action(tr("Restore the default server", "恢复默认服务器"), tr("Back to the list it comes with: ", "恢复预置服务器列表：") + String.join(", ", DEFAULT_SERVERS) + ".",
                () -> {
                    new File(folder, "hub_servers").delete();
                    build();
                });

        header(tr("Direct connections", "直接连接"));
        List<String> hosts = lines("hub_hosts");
        for (int i = 0; i < hosts.size(); i++) {
            final int k = i;
            row(hosts.get(i), tr("Remove", "移除"), tr("Tap to remove this host.", "点按移除此主机。"), v -> {
                List<String> l = lines("hub_hosts");
                if (k < l.size()) l.remove(k);
                writeLines("hub_hosts", l);
                build();
            }, false);
        }
        action(tr("Add search host", "添加搜索主机"), tr("Another phone's address (IP, or IP:port) to look for its hub directly, on the same "
                + "Wi-Fi or with its port forwarded.", "输入另一台设备的 IP 或 IP:端口以搜索其集会区域。两台设备需在同一 Wi-Fi 下，或已配置端口转发。"), () -> ask(tr("Add search host", "添加搜索主机"), tr("IP or IP:port", "IP 或 IP:端口"), "", 128, text -> {
                    if (text.isEmpty()) return;
                    List<String> l = lines("hub_hosts");
                    if (!l.contains(text)) l.add(text);
                    writeLines("hub_hosts", l);
                }));
        List<String> port = lines("hub_port");
        String portNow = port.isEmpty() ? "39510" : port.get(0);
        row(tr("Port", "端口"), portNow.equals("0") ? tr("Off", "关") : portNow,
                tr("The TCP port a hub you create listens on for direct connections (0: off). Hub servers don't need it.", "直连时本机集会区域监听的 TCP 端口，0 表示关闭。使用集会区域服务器时无需设置。"),
                v -> ask(tr("Port", "端口"), tr("1-65535, or 0 for off", "1 至 65535，0 表示关闭"), portNow, 5, text -> {
                    try {
                        int p = Integer.parseInt(text);
                        if (p >= 0 && p <= 65535) writeLines("hub_port", Arrays.asList(Integer.toString(p)));
                    } catch (NumberFormatException ignored) {
                        // (not a number: unchanged)
                    }
                }), false);
    }

    // Switches for testing drivers and collecting logs. Same files as always (see the options table in the README), just
    // with a name here. Most of them are only read at start-up.
    private static native byte[] nativeLuaReport();

    // What the Lua runtime did since the game started, in the settings menu's own look: where it stands, then its log
    // (scripts loaded or failed, reloads), errors in red
    private void showLuaReport() {
        String text = new String(nativeLuaReport(), java.nio.charset.StandardCharsets.UTF_8).trim();
        String state;
        switch (ModUi.runtime()) {
            case 2: state = tr("Running: the scripts are loaded.", "运行中：脚本已加载。"); break;
            case 1: state = tr("Ready: the scripts load once you're in the village or a quest.", "已就绪：进入村庄或任务后加载脚本。"); break;
            default: state = tr("Not running: no Lua scripts were installed when the game started.", "未运行：游戏启动时没有安装 Lua 脚本。");
        }
        final Dialog d = new Dialog(activity);
        d.requestWindowFeature(Window.FEATURE_NO_TITLE);
        LinearLayout box = new LinearLayout(activity);
        box.setOrientation(LinearLayout.VERTICAL);
        box.setPadding(dp(16), dp(12), dp(16), dp(14));
        box.setBackground(plate(PANEL, 2, 0));
        TextView title = text("Lua / TDB", 20, GOLD, true);
        title.setTypeface(Typeface.create(Typeface.SERIF, Typeface.BOLD));
        box.addView(title);
        box.addView(rule(RED, 2));
        box.addView(rule(GREEN, 1));
        TextView now = text(state, 16, CREAM, false);
        now.setPadding(0, dp(12), 0, dp(8));
        box.addView(now);

        LinearLayout lines = new LinearLayout(activity);
        lines.setOrientation(LinearLayout.VERTICAL);
        lines.setPadding(dp(10), dp(8), dp(10), dp(8));
        lines.setBackground(plate(ROW, 1, 0));
        if (text.isEmpty()) lines.addView(text(tr("Nothing logged yet.", "暂无记录。"), 14, DIM, false));
        for (String line : text.split("\n")) {
            if (line.isEmpty()) continue;
            boolean bad = line.contains("rror") || line.contains("failed") || line.contains(" hook: ");
            TextView t = text(line, 13, bad ? 0xFFF08070 : CREAM, false);
            t.setTypeface(Typeface.MONOSPACE);
            t.setTextIsSelectable(true);
            t.setPadding(0, dp(2), 0, dp(2));
            lines.addView(t);
        }
        ScrollView scroll = new ScrollView(activity);
        scroll.addView(lines);
        DisplayMetrics m = activity.getResources().getDisplayMetrics();
        box.addView(scroll, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));

        LinearLayout buttons = new LinearLayout(activity);
        buttons.setPadding(0, dp(14), 0, 0);
        TextView copy = button(tr("Copy", "复制"), false);
        copy.setOnClickListener(v -> {
            android.content.ClipboardManager clip =
                    (android.content.ClipboardManager) activity.getSystemService(android.content.Context.CLIPBOARD_SERVICE);
            if (clip != null) clip.setPrimaryClip(android.content.ClipData.newPlainText("Lua / TDB", state + "\n" + text));
            d.dismiss();
        });
        TextView ok = button(tr("OK", "确定"), true);
        ok.setOnClickListener(v -> d.dismiss());
        LinearLayout.LayoutParams half = new LinearLayout.LayoutParams(0, dp(48), 1);
        half.setMargins(dp(4), 0, dp(4), 0);
        buttons.addView(copy, half);
        buttons.addView(ok, new LinearLayout.LayoutParams(half));
        box.addView(buttons);

        d.setContentView(box);
        Window w = d.getWindow();
        if (w != null) {
            w.setBackgroundDrawable(new ColorDrawable(Color.TRANSPARENT));
            // (as tall as the log needs, the lines scrolling past 80% of the screen)
            box.measure(View.MeasureSpec.makeMeasureSpec(Math.min(dp(720), (int) (m.widthPixels * 0.8f)), View.MeasureSpec.EXACTLY),
                    View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED));
            int tall = Math.min(box.getMeasuredHeight(), (int) (m.heightPixels * 0.8f));
            w.setLayout(Math.min(dp(720), (int) (m.widthPixels * 0.8f)), tall);
        }
        d.show();
    }

    private void modChanged(File dir) {
        modChanged(ModFiles.gameFiles(dir), ModFiles.scripts(dir));
    }

    private void modChanged(boolean gameFiles, boolean scripts) {
        if (!activity.modsChanged(gameFiles, scripts) && scripts)
            android.widget.Toast.makeText(activity, tr("Script reload requested.", "已请求重新加载 Lua 脚本。"), android.widget.Toast.LENGTH_SHORT).show();
    }

    // Game files changed: restart now (losing what isn't saved), or later from Mods & cheats
    void askRestart() {
        dropDown(tr("Restart to load the mod's game files? Progress since your last save is lost.",
                        "重启以加载 MOD 的游戏资源？上次存档之后的进度将会丢失。"),
                new String[] {tr("Later", "稍后"), tr("Restart now", "立即重启")}, -1, k -> {
                    if (k == 1) activity.restartGame();
                });
    }

    private void buildDebug() {
        if (GameActivity.luaBuild(activity)) {
            action("Lua / TDB", tr("View startup script results.", "查看启动时脚本的运行结果。"), this::showLuaReport);
            flag(tr("Object Explorer", "对象浏览器"),
                    tr("REFramework's explorer in the MOD UI: search the game's types, see their fields and methods, and "
                            + "read a manager's fields live. For writing scripts. Takes effect on the next start.",
                            "在 MOD 界面中加入 REFramework 的对象浏览器：搜索游戏类型、查看字段和方法，并实时读取管理器的字段。用于编写脚本。下次启动生效。"),
                    "object_explorer", false);
        }
        String next = tr(" Takes effect on the next start.", "下次启动生效。");
        note(tr("For testing drivers and sending logs. If the game looks right, leave all of this as it is.", "用于测试驱动和收集日志。画面正常时可保留默认设置。"));
        header(tr("Log", "日志"));
        flag(tr("Detailed log", "详细日志"),
                tr("Writes much more to log.txt (every draw's state, textures…). Slower: only for a report.", "将每次绘制的状态、纹理等写入 log.txt，会降低性能。仅建议在收集报告时开启。"),
                "verbose_log", false);
        flag(tr("GPU breadcrumbs", "GPU 执行标记"),
                tr("Logs which pass the GPU stops at when the driver drops work (black screen). Slower.", "在驱动丢弃工作、画面变黑时记录 GPU 停在哪个渲染阶段。会降低性能。") + next,
                "breadcrumbs", false);
        action(tr("Profile one GPU frame", "分析一帧 GPU 耗时"),
                tr("Writes how long each pass of one frame takes to the log (\"perfil GPU\" lines). Use it where it's slow.", "将一帧各渲染阶段的耗时写入日志的「perfil GPU」行。请在性能较差的位置使用。"),
                () -> {
                    trigger("profile_gpu", "");
                    dialog.dismiss();
                });
        flag(tr("Performance stats", "性能统计"),
                tr("Every 10 seconds, where the port's CPU time goes (\"perfil\" lines in the log), without the rest of "
                        + "the detailed log. Turn it on, play a minute, and send the report.", "每 10 秒将 CPU 耗时分布写入日志的「perfil」行。开启后游玩一分钟再保存报告即可。"),
                "perf_stats", false);
        header(tr("Bug report", "问题报告"));
        note(tr("For testers: stand where the problem shows, tap Capture this frame, wait a few seconds, then Save the "
                + "report and send the zip from your Download folder.", "在能看到问题的位置点「捕获当前帧」，稍等几秒，再保存报告。压缩包位于下载目录。") + (reportStatus != null ? "\n" + reportStatus : ""));
        action(tr("Capture this frame", "捕获当前帧"),
                tr("Saves the next frame's pictures (every render target, depth and shadow maps) and a trace of its draws "
                        + "to the log. The game stutters for a moment. Old captures are cleared first.", "保存下一帧的渲染目标、深度、阴影纹理及绘制记录。捕获时会短暂停顿，并先清理旧捕获。"),
                () -> {
                    BugReport.capture(folder);
                    reportStatus = tr("Frame captured. Wait a few seconds before saving the report.", "已请求捕获，请稍等几秒再保存报告。");
                    dialog.dismiss();
                });
        action(tr("Capture loading illustration", "捕获加载插画"),
                tr("Saves draw states and pixels before scaling for the next frame that binds the Rise loading illustration, in a loading-f… folder under the game data folder.", "捕获下一次绑定加载插画时的绘制状态及缩放前像素，保存至游戏数据目录下的 loading-f… 文件夹。"),
                () -> {
                    reportStatus = BugReport.trigger(folder, "trazar", "loading_iam")
                            ? tr("Loading illustration capture requested.", "已请求捕获加载插画。") : tr("Could not write the capture request.", "无法写入捕获请求。");
                    build();
                });
        action(tr("Save the report", "保存报告"),
                tr("Puts log.txt, your settings and the captured frame into Download/mhrise-report.zip, to send.", "将 log.txt、设置及已捕获的帧保存到 Download/mhrise-report.zip。"),
                this::saveReport);
        flag(tr("Capture button", "悬浮捕获按钮"),
                tr("A small round button over the game, so a frame can be captured right where the problem shows without "
                        + "opening this menu. Tap it to capture, hold it to save the report, drag it out of the way.", "轻点悬浮按钮可捕获当前帧，长按可保存报告，拖动可调整位置。"),
                CaptureButton.SWITCH, false);
        header(tr("Shaders", "着色器"));
        flag(tr("Remove unused shader functions", "移除未使用的着色器函数"),
                tr("Strips shader code that's never called before the driver sees it. Qualcomm's driver (512.863) "
                        + "rejected the character pipelines because of it (they came out black).", "在交给驱动前移除未调用的着色器代码。高通 512.863 驱动曾因这些代码拒绝角色管线，导致角色变黑。") + next,
                "keep_dead_functions", true);
        toggle(tr("Integer vertex attributes", "整数顶点属性"),
                tr("Integer vertex data (bone indices…) reaches the shader as integers, as on the console. Without "
                        + "it, characters stretch into spikes on Mali GPUs. Off by default on Adreno, which doesn't need "
                        + "it (with Turnip it made characters flash).", "将骨骼索引等整数顶点数据以整数传给着色器。关闭可能导致 Mali 上角色拉伸成尖刺。Adreno 默认关闭，部分 Turnip 驱动开启后会导致角色闪烁。") + next,
                () -> (gpu.adreno() ? has("int_attribs") : !has("no_int_attribs")) ? 1 : 0, v -> {
                    put("int_attribs", gpu.adreno() && v != 0);
                    put("no_int_attribs", !gpu.adreno() && v == 0);
                });
        flag(tr("spirv-opt on switch shaders", "对含 switch 的着色器运行 spirv-opt"),
                tr("Runs shaders that contain a switch (BRX) through the optimizer; some drivers miscompile them.", "优化包含 switch（BRX）的着色器，解决部分驱动的编译问题。") + next,
                "no_spirv_opt_brx", true);
        flag(tr("Retry rejected pipelines", "重试被拒绝的管线"),
                tr("When the driver rejects a pipeline, its shaders are translated again with safer options.", "驱动拒绝管线时，使用更保守的选项重新翻译着色器。"),
                "no_safe_translation", true);
        exclusive(tr("GPU occlusion culling", "GPU 遮挡剔除"),
                tr("The game hides what its GPU test says is behind something else. Auto: off on Mali, where that test's "
                        + "compute shaders don't run right on 16-lane subgroups and grass and trees popped in and out; on "
                        + "elsewhere. Off: the game culls by the camera alone, a bit more to draw. Applies at once.", "隐藏 GPU 判定被遮挡的物体。自动模式在 Mali 上关闭，避免 16 通道子组计算异常导致草木闪现；其他设备默认开启。关闭后仅按相机范围剔除。立即生效。"),
                new String[] {tr("Auto", "自动"), tr("On", "开"), tr("Off", "关")}, new String[] {null, "gpu_occlusion", "no_gpu_occlusion"});
        if (gpu.mali())
            exclusive(tr("Exact draw counts (old drivers)", "精确绘制计数（旧驱动）"),
                    tr("On drivers without draw-indirect-count (ARM's on a Dimensity 720), the vertex shader drops the "
                            + "game's GPU-counted draws past their count, which removes the stretched triangles. Auto: on "
                            + "where the driver lacks draw-indirect-count, unused elsewhere.", "在缺少 draw-indirect-count 的驱动上，由顶点着色器丢弃超出实际计数的绘制，避免拉长的三角形。自动模式仅在缺少该功能时启用。") + next,
                    new String[] {tr("Auto", "自动"), tr("On", "开"), tr("Off", "关")}, new String[] {null, "count_check", "no_count_check"});
        if (gpu.mali())
            exclusive(tr("Subgroups under 32", "小于 32 的子组"),
                    tr("Subgroup operations on GPUs whose subgroups are under 32 (Mali: 16). Auto: on with PanVK and with "
                            + "ARM's driver on GPUs without draw-indirect-count (Dimensity 720), where the ground goes "
                            + "missing without them; off with newer ARM drivers, where they stalled the GPU. Quad swaps for the game's derivatives stay on "
                            + "either way.", "在小于 32 通道的子组上使用子组操作。自动模式对 PanVK 及缺少 draw-indirect-count 的 ARM 驱动启用，以避免地面消失；新版 ARM 驱动上关闭以防 GPU 停滞。计算导数用的四元组交换始终启用。") + next,
                    new String[] {tr("Auto", "自动"), tr("On", "开"), tr("Off", "关")}, new String[] {null, "short_subgroups", "no_short_subgroups"});
        header(tr("Driver workarounds", "驱动兼容选项"));
        if (gpu.adreno()) {
            note(tr("Render pass shortcuts: for flicker that only shows without Turnip's system memory rendering. Turn "
                    + "on one switch at a time to find which one it is. Each costs a little speed. Applies at once.", "用于排查 Turnip 分块渲染时的闪烁。可逐项开启检查，每项都会增加少许开销。立即生效。"));
            flag(tr("No clears in the pass", "独立执行清除"),
                    tr("A clear the game asks for is done on its own, not folded into the start of the next render pass.", "游戏请求清除时单独执行，不并入下一次渲染通道的开头。"),
                    "no_deferred_clear", false);
            flag(tr("Save every image", "保存所有图像"),
                    tr("Everything a render pass draws is written back, even what the port thinks isn't read again.", "将每次渲染的结果写回，即使预计不会再次读取也保存。"),
                    "no_store_ops", false);
            flag(tr("Full render area", "使用完整渲染区域"),
                    tr("Each render pass covers the whole image, not just the part that's drawn.", "每个渲染通道覆盖整张图像。"),
                    "no_area", false);
            flag(tr("Qualcomm: return after a switch", "高通：switch 后返回"),
                    tr("On Qualcomm's driver, a return on a switch's impossible path instead of \"unreachable\" (without "
                            + "it the hunter's face wasn't drawn).", "在高通驱动上，switch 不可达的分支使用 return，避免猎人面部不显示。") + next,
                    "no_return_qualcomm", true);
            flag(tr("Qualcomm: emulated shuffle", "高通：模拟 shuffle"),
                    tr("On Qualcomm's driver, shuffle with a dynamic index is emulated, as in Eden.", "在高通驱动上模拟动态索引的 shuffle 操作。") + next,
                    "no_shuffle_qualcomm", true);
        }
        if (gpu.mali()) {
            flag(tr("Mali: return after a switch", "Mali：switch 后返回"),
                    tr("The same on Mali GPUs (ARM's driver and PanVK).", "在 ARM 和 PanVK 驱动上使用相同的返回处理。") + next,
                    "no_return_mali", true);
            flag(tr("Mali: safe buffer reads", "Mali：安全缓冲区读取"),
                    tr("A read past the end of a buffer gives 0 instead of a GPU fault, which on Mali throws away the "
                            + "frame's work (black or magenta screens that fix themselves when you leave and come back). "
                            + "Costs very little.", "越界读取返回 0，避免 GPU 故障导致整帧被丢弃、画面变黑或品红。性能开销很小。") + next,
                    "no_robust", true);
        }

        header(tr("Driver environment", "驱动环境变量"));
        note(tr("Settings for the GPU driver itself, as NAME=value, set before it starts (on the next start). Mesa's drivers "
                + "read them: Turnip (TU_DEBUG, FD_DEV_FEATURES) and PanVK (PANVK_DEBUG). The phone's own drivers "
                + "ignore them.", "按 NAME=value 格式设置，下次驱动启动前生效。Mesa 驱动支持 TU_DEBUG、FD_DEV_FEATURES 或 PANVK_DEBUG 等变量，设备原厂驱动通常忽略。"));
        if (turnip(activity)) {  // (Turnip's own variables, only while a Turnip driver is the one chosen)
            envToken(tr("Turnip: no LRZ", "Turnip：禁用 LRZ"), tr("TU_DEBUG=nolrz. Turns off Turnip's low-resolution depth test, which culls whole "
                    + "blocks of the screen. Try it if blocks of the picture flash black.", "TU_DEBUG=nolrz。关闭按块剔除的低分辨率深度测试，可用于排查画面局部黑闪。"), "TU_DEBUG", "nolrz");
            envToken(tr("Turnip: system memory rendering", "Turnip：系统内存渲染"), tr("TU_DEBUG=sysmem. Draws straight to memory instead of in the "
                    + "GPU's on-chip tiles. Slower; for flicker or flashing on new Adreno GPUs.", "TU_DEBUG=sysmem。直接渲染到系统内存，通常较慢，可用于排查新款 Adreno 上的闪烁。"), "TU_DEBUG", "sysmem");
            envToken(tr("Turnip: no UBWC", "Turnip：禁用 UBWC"), tr("TU_DEBUG=noubwc. No compression on any image.", "TU_DEBUG=noubwc。关闭所有图像的 UBWC 压缩。"), "TU_DEBUG", "noubwc");
            envToken(tr("Turnip: UBWC flag hint", "Turnip：UBWC 标志提示"), tr("FD_DEV_FEATURES=enable_tp_ubwc_flag_hint=1. Reported to stop screen "
                    + "flicker with some Adreno 8xx Turnip builds.", "FD_DEV_FEATURES=enable_tp_ubwc_flag_hint=1。部分 Adreno 8xx 的 Turnip 版本可借此消除闪烁。"), "FD_DEV_FEATURES", "enable_tp_ubwc_flag_hint=1");
        } else {
            note(tr("Turnip's switches show here while a Turnip driver is chosen (Graphics → GPU driver).", "选择 Turnip 驱动后，此处会显示它的选项。"));
        }
        List<String> env = ini.env();
        for (int i = 0; i < env.size(); i++) {
            final int k = i;
            row(env.get(i), tr("Remove", "移除"), tr("Tap to remove this variable.", "点按移除此变量。"), v -> {
                List<String> l = ini.env();
                if (k < l.size()) l.remove(k);
                ini.setEnv(l);
                build();
            }, false);
        }
        action(tr("Add variable", "添加变量"), tr("Any driver variable, as NAME=value.", "按 NAME=value 格式输入驱动环境变量。"),
                () -> ask(tr("Add variable", "添加变量"), "NAME=value", "", 256, text -> {
                    int eq = text.indexOf('=');
                    if (eq <= 0) return;
                    setEnv(text.substring(0, eq).trim(), text.substring(eq + 1).trim());
                }));
    }

    // ---- driver environment ("env.NAME" lines of mhr.ini, read by vk.cpp at start-up) -----------------------------------

    // Is the driver chosen in DriverActivity a Turnip (Mesa's freedreno library, or a folder named for it)?
    static boolean turnip(android.content.Context c) {
        try {
            List<String> l = Files.readAllLines(new File(new File(c.getFilesDir(), "drivers"), "elegido").toPath());
            if (l.size() < 2) return false;
            String folder = l.get(0).toLowerCase(java.util.Locale.ROOT), library = l.get(1).toLowerCase(java.util.Locale.ROOT);
            return library.contains("freedreno") || folder.contains("turnip");
        } catch (IOException | RuntimeException e) {
            return false;  // (the system's driver)
        }
    }

    private String envValue(String name) {
        for (String l : ini.env())
            if (l.startsWith(name + "=")) return l.substring(name.length() + 1);
        return null;
    }

    private void setEnv(String name, String value) {
        List<String> l = ini.env();
        l.removeIf(x -> x.startsWith(name + "="));
        if (value != null && !value.isEmpty()) l.add(name + "=" + value);
        ini.setEnv(l);
    }

    // One option of a comma-separated variable (TU_DEBUG=nolrz,sysmem), or a whole NAME=value, as a toggle
    private void envToken(String title, String help, String name, String token) {
        final boolean list = !token.contains("=");
        toggle(title, help, () -> {
            String v = envValue(name);
            if (v == null) return 0;
            return (list ? Arrays.asList(v.split(",")).contains(token) : v.equals(token)) ? 1 : 0;
        }, on -> {
            String v = envValue(name);
            if (!list) {
                setEnv(name, on != 0 ? token : null);
            } else {
                List<String> parts = new ArrayList<>();
                if (v != null) for (String p : v.split(",")) if (!p.isEmpty() && !p.equals(token)) parts.add(p);
                if (on != 0) parts.add(token);
                setEnv(name, String.join(",", parts));
            }
            build();
        });
    }

    private void buildTouch() {
        header(tr("Touch controls", "触控"));
        toggle(tr("On-screen controls", "显示虚拟手柄"),
                tr("The buttons and sticks over the game. Off, only the gear stays, to come back here.", "在游戏画面上显示按键和摇杆。关闭后保留齿轮按钮，可从那里重新打开设置。"),
                () -> cfg.touchOn ? 1 : 0, v -> cfg.touchOn = v != 0);
        number(tr("Opacity", "不透明度"), tr("How much of the game shows through the controls.", "调整透过控件看到游戏画面的程度。"), 20, 100, 5, " %", () -> cfg.opacity, v -> cfg.opacity = v);
        number(tr("Size", "大小"), tr("All controls at once. Each one also has its own, in Edit layout.", "统一调整所有控件大小。可在「编辑布局」中单独调整各个控件。"), 60, 160, 5, " %", () -> cfg.size,
                v -> cfg.size = v);
        choice(tr("Vibration", "振动"), tr("A short tap when a control is pressed.", "按下控件时给予短暂的振动反馈。"),
                new String[] {tr("Off", "关"), tr("Light", "弱"), tr("Medium", "中"), tr("Strong", "强")},
                () -> cfg.vibration, v -> {
                    cfg.vibration = v;
                    activity.tick();
                });
        toggle(tr("Slide between buttons", "滑动切换按键"),
                tr("A finger moving from one button to another lets go of the first and presses the second, without lifting.", "手指从一个按键滑到另一个时，松开前一个并按下后一个，无需抬手。"),
                () -> cfg.slide ? 1 : 0, v -> cfg.slide = v != 0);
        toggle(tr("Camera swipe", "滑动转动视角"),
                tr("Dragging a finger on the game, away from the controls, turns the camera, as in a phone game. A tap still "
                        + "goes to the game's touch screen. If you'd rather use the right stick, show it below and turn "
                        + "this off.", "在控件以外的区域拖动可转动镜头；轻点仍传给游戏触屏。也可以在布局中显示右摇杆并关闭此项。"),
                () -> cfg.cameraSwipe ? 1 : 0, v -> {
                    cfg.cameraSwipe = v != 0;
                    build();  // (the camera warning at the end of the page)
                });
        number(tr("Swipe speed", "滑动速度"), tr("How far the camera turns for the same drag.", "相同拖动距离对应的镜头转动幅度。"), 25, 300, 25, " %", () -> cfg.swipeSpeed,
                v -> cfg.swipeSpeed = v);
        number(tr("Stick dead zone", "摇杆死区"),
                tr("How far a touch stick moves from its centre before it counts.", "虚拟摇杆离开中心多远后开始响应。"), 0, 30, 1, " %",
                () -> cfg.touchDeadzone, v -> cfg.touchDeadzone = v);
        toggle(tr("Tap a stick: L3 / R3", "轻点摇杆触发 L3 / R3"),
                tr("A short tap on a stick, without moving it, presses it (L3 left, R3 right). Their own buttons are there too.", "轻点未移动的左、右摇杆可分别触发 L3、R3，也可使用独立按键。"),
                () -> cfg.stickPress ? 1 : 0, v -> cfg.stickPress = v != 0);
        action(tr("Edit layout", "编辑布局"),
                tr("Drag each control where you want it. At the top: its size, hide or show it, and back to the original.", "拖动控件以调整位置。顶部可更改大小、隐藏或显示控件，以及恢复默认布局。"),
                () -> {
                    dialog.dismiss();
                    activity.startEditing();
                });
        action(tr("Export layout", "导出布局"), tr("Saves the whole touch layout to a file: where each control is, its look and pictures, "
                + "and the combos with their controller buttons. To keep it, or give it to someone.", "将所有控件的位置、外观、图片，以及组合键和实体手柄绑定保存到文件。"), () -> {
                    dialog.dismiss();
                    activity.exportLayout();
                });
        action(tr("Import layout", "导入布局"), tr("Replaces the touch layout with one from an exported file (from this phone or another: "
                + "places follow the screen).", "用导出的文件替换当前触控布局。可从其他设备导入，位置会按屏幕比例调整。"), () -> {
                    dialog.dismiss();
                    activity.importLayout();
                });
        action(tr("Reset layout", "重置布局"),
                tr("The traditional layout, every control in its place and at its size.", "将所有控件恢复到默认位置与大小。"),
                () -> {
                    cfg.resetLayout();
                    changed();
                    build();
                });

        // (showing and hiding each control is in Edit layout: Hide / Show on the chosen one)
        if (!cfg.shown[Controls.Ctl.RSTICK.ordinal()] && !cfg.cameraSwipe)
            note(tr("The right stick is hidden and camera swipe is off: nothing on the screen turns the camera. Show the stick "
                    + "in Edit layout, or turn camera swipe on.", "右摇杆已隐藏，滑动视角也已关闭。若要用触屏转动镜头，请在编辑布局中显示右摇杆，或开启滑动视角。"));
    }

    private void buildCombos() {
        header(tr("Combos", "组合键"));
        note(tr("A combo presses several buttons at once: with its button on the screen, or with one (or two) of the controller's. "
                + "The layout editor's Add and Copy as a new button make combos too, so every added button can be fired "
                + "from a controller.", "组合键可同时按下多个按键，用触屏或手柄上的一至两个按键触发。布局编辑器中添加或复制的按键也可在此绑定。"));
        for (int i = 0; i < Controls.COMBOS; i++) {
            final int k = i;
            String value = cfg.combo[k] == 0 ? tr("None", "无")
                                             : Controls.names(cfg.combo[k]) + (cfg.comboPad[k] != 0
                                                       ? tr("  ·  pad: ", "  ·  手柄：") + Controls.names(cfg.comboPad[k]) : "")
                                                       + (cfg.comboKey[k] != 0 ? tr("  ·  key: ", "  ·  键位：") + Controls.keyName(cfg.comboKey[k]) : "");
            row(cfg.comboName(k), value, tr("Choose the buttons it presses and, if you like, the controller button that does it.", "选择要同时按下的按键，也可指定触发它的实体按键。"), v -> editCombo(k), false);
        }

    }

    private void buildController() {
        header(tr("Controller", "实体手柄"));
        number(tr("Dead zone", "死区"), tr("How far a controller stick moves before it counts (both sticks).", "实体手柄左右摇杆离开中心多远后开始响应。"), 0, 40, 1, " %",
                () -> cfg.padDeadzone, v -> cfg.padDeadzone = v);
        number(tr("Analog triggers", "模拟扳机"),
                tr("How far an analog trigger travels before it presses ZL / ZR.", "实体扳机按下多少行程后触发 ZL / ZR。"), 10, 90, 5, " %",
                () -> cfg.trigger, v -> cfg.trigger = v);
        choice(tr("Right stick", "右摇杆"),
                tr("Which axes the right stick is read from. By controller: Z/RZ on Android and Xbox pads, RX/RY (with the "
                        + "triggers on Z/RZ) on many generic ones. Change it if the camera won't move or moves on its own.", "选择读取右摇杆的轴。自动模式下，Android 和 Xbox 手柄使用 Z/RZ，部分通用手柄使用 RX/RY。镜头不动或自行移动时可调整。"),
                new String[] {tr("By controller", "按手柄识别"), "Z / RZ", "RX / RY"}, () -> cfg.rightAxes,
                v -> cfg.rightAxes = v);

        header(tr("Button mapping", "按键映射"));
        note(tr("Tap a Switch button, then press the controller button you want for it. Everything else keeps its usual "
                + "place. To swap two buttons, map each one to the other.", "点按一个 Switch 按键，再按要绑定的实体手柄按键。要交换两个键，需分别映射。"));
        for (int i = 0; i < Controls.BUTTONS.length; i++) {
            final int b = i;
            String value = cfg.mapKey[i] == 0 ? tr("Default", "默认") : Controls.keyName(cfg.mapKey[i]);
            row(mapNames[i], value, tr("The controller button that presses ", "选择用于触发游戏中 ") + mapNames[i] + tr(" in the game.", "的实体按键。"),
                    v -> captureKey(b), false);
        }
        row(tr("Reset mapping", "重置映射"), "", tr("Every controller button back to its usual place.", "恢复所有实体手柄按键的默认映射。"), v -> {
            Arrays.fill(cfg.mapKey, 0);
            changed();
            build();
        }, false);

        header(tr("Added buttons and combos", "自定义按键和组合键"));
        note(tr("Any key of a controller or a keyboard can fire one of your added buttons or combos (Combos, and the layout "
                + "editor's Add): while it's held the game gets all of its buttons. Tap one, then press the key.", "可以用手柄或键盘的任意键触发已添加的按键或组合键。点选一项后按下要绑定的键。"));
        boolean any = false;
        for (int i = 0; i < Controls.COMBOS; i++) {
            if (cfg.combo[i] == 0) continue;
            any = true;
            final int k = i;
            final String name = cfg.comboName(k);
            row(name + "  (" + Controls.names(cfg.combo[k]) + ")", cfg.comboKey[k] == 0 ? tr("None", "无") : Controls.keyName(cfg.comboKey[k]),
                    tr("The controller or keyboard key that presses ", "选择用来触发此组合的手柄键或键盘键：") + Controls.names(cfg.combo[k]) + ".",
                    v -> captureKey(name, cfg.comboKey[k], key -> {
                        for (int j = 0; j < Controls.COMBOS; j++)
                            if (cfg.comboKey[j] == key) cfg.comboKey[j] = 0;  // (one key, one of them)
                        cfg.comboKey[k] = key;
                    }), false);
        }
        if (!any) note(tr("None yet: make one in Combos, or with Add in Edit layout.", "尚无自定义按键。可在「组合键」或「编辑布局」中添加。"));
    }

    private String[] mappingNames() {
        return new String[] {"A", "B", "X", "Y", "L", "R", "ZL", "ZR", tr("+ (Plus)", "+（加号）"), tr("− (Minus)", "−（减号）"),
        tr("L3 (left stick press)", "L3（按下左摇杆）"), tr("R3 (right stick press)", "R3（按下右摇杆）"), tr("D-pad up", "方向键上"), tr("D-pad down", "方向键下"), tr("D-pad left", "方向键左"), tr("D-pad right", "方向键右"),
        tr("Left stick up", "左摇杆向上"), tr("Left stick down", "左摇杆向下"), tr("Left stick left", "左摇杆向左"), tr("Left stick right", "左摇杆向右"), tr("Right stick up", "右摇杆向上"), tr("Right stick down", "右摇杆向下"),
        tr("Right stick left", "右摇杆向左"), tr("Right stick right", "右摇杆向右")};
    }

    // ---- button mapping -------------------------------------------------------------------------------------------------

    // Waits for one controller button (or an analog trigger, or a hat direction) and maps it to Controls.BUTTONS[b]
    private void captureKey(int b) {
        captureKey(mapNames[b], cfg.mapKey[b], key -> {
            for (int i = 0; i < cfg.mapKey.length; i++)
                if (cfg.mapKey[i] == key) cfg.mapKey[i] = 0;  // (one controller button, one Switch button)
            cfg.mapKey[b] = key;
        });
    }

    // The same for anything a key is mapped to: set gets the key pressed, or 0 for Default (none)
    private void captureKey(String what, int current, IntConsumer set) {
        final Dialog d = new Dialog(activity);
        d.requestWindowFeature(Window.FEATURE_NO_TITLE);
        LinearLayout box = new LinearLayout(activity);
        box.setOrientation(LinearLayout.VERTICAL);
        box.setPadding(dp(16), dp(12), dp(16), dp(14));
        box.setBackground(plate(PANEL, 2, 0));
        TextView title = text(tr("Map ", "映射 ") + what, 20, GOLD, true);
        title.setTypeface(Typeface.create(Typeface.SERIF, Typeface.BOLD));
        box.addView(title);
        box.addView(rule(RED, 2));
        box.addView(rule(GREEN, 1));
        TextView ask = text(tr("Press the controller button or keyboard key for ", "请按下要绑定到以下功能的实体手柄或键盘按键：") + what + ".", 16, CREAM, false);
        ask.setPadding(0, dp(12), 0, dp(4));
        box.addView(ask);
        box.addView(text(current == 0 ? tr("Now: none of its own", "当前未绑定") : tr("Now: ", "当前：") + Controls.keyName(current), 14, DIM, false));

        final boolean[] done = {false};
        final IntConsumer take = key -> {
            if (done[0]) return;
            done[0] = true;
            set.accept(key);
            d.dismiss();
            changed();
            build();
        };
        d.setOnKeyListener((dialog, keyCode, ev) -> {
            // (a controller's, or a physical keyboard's: not the on-screen one's)
            boolean pad = (ev.getSource() & (InputDevice.SOURCE_GAMEPAD | InputDevice.SOURCE_JOYSTICK
                    | InputDevice.SOURCE_DPAD | InputDevice.SOURCE_KEYBOARD)) != 0 && ev.getDevice() != null
                    && !ev.getDevice().isVirtual();
            if (keyCode == KeyEvent.KEYCODE_VOLUME_UP || keyCode == KeyEvent.KEYCODE_VOLUME_DOWN) return false;
            if (!pad) return keyCode != KeyEvent.KEYCODE_BACK;  // (the back gesture still closes it)
            if (ev.getAction() == KeyEvent.ACTION_UP && ev.getRepeatCount() == 0) take.accept(keyCode);
            return true;  // (on release, so the press doesn't also reach the menu)
        });
        // Analog triggers and hats send motion, not keys: they're mapped as the keys they stand for
        box.setFocusable(true);
        box.setFocusableInTouchMode(true);
        box.setOnGenericMotionListener((v, ev) -> {
            if ((ev.getSource() & InputDevice.SOURCE_JOYSTICK) == 0) return false;
            float lt = Math.max(ev.getAxisValue(MotionEvent.AXIS_LTRIGGER), ev.getAxisValue(MotionEvent.AXIS_BRAKE));
            float rt = Math.max(ev.getAxisValue(MotionEvent.AXIS_RTRIGGER), ev.getAxisValue(MotionEvent.AXIS_GAS));
            float hx = ev.getAxisValue(MotionEvent.AXIS_HAT_X), hy = ev.getAxisValue(MotionEvent.AXIS_HAT_Y);
            if (lt > 0.6f) take.accept(KeyEvent.KEYCODE_BUTTON_L2);
            else if (rt > 0.6f) take.accept(KeyEvent.KEYCODE_BUTTON_R2);
            else if (hy < -0.5f) take.accept(KeyEvent.KEYCODE_DPAD_UP);
            else if (hy > 0.5f) take.accept(KeyEvent.KEYCODE_DPAD_DOWN);
            else if (hx < -0.5f) take.accept(KeyEvent.KEYCODE_DPAD_LEFT);
            else if (hx > 0.5f) take.accept(KeyEvent.KEYCODE_DPAD_RIGHT);
            return true;
        });

        LinearLayout buttons = new LinearLayout(activity);
        buttons.setPadding(0, dp(14), 0, 0);
        TextView usual = button(tr("Default", "默认"), false);
        usual.setOnClickListener(v -> {
            done[0] = true;
            set.accept(0);
            d.dismiss();
            changed();
            build();
        });
        TextView cancel = button(tr("Cancel", "取消"), true);
        cancel.setOnClickListener(v -> {
            done[0] = true;
            d.dismiss();
        });
        LinearLayout.LayoutParams half = new LinearLayout.LayoutParams(0, dp(46), 1);
        half.setMargins(dp(4), 0, dp(4), 0);
        buttons.addView(usual, half);
        buttons.addView(cancel, new LinearLayout.LayoutParams(half));
        box.addView(buttons);

        d.setContentView(box);
        Window w = d.getWindow();
        if (w != null) {
            w.setBackgroundDrawable(new ColorDrawable(Color.TRANSPARENT));
            DisplayMetrics m = activity.getResources().getDisplayMetrics();
            w.setLayout(Math.min(dp(520), (int) (m.widthPixels * 0.7f)), ViewGroup.LayoutParams.WRAP_CONTENT);
        }
        d.show();
        box.requestFocus();
    }

    private String hotkeyName(int[] keys) {
        if (keys.length == 0) return tr("none", "无");
        StringBuilder b = new StringBuilder();
        for (int k : keys) b.append(b.length() > 0 ? " + " : "").append(Controls.keyName(k));
        return b.toString();
    }

    // Waits for one to three controller buttons held together; the chord is taken when they're all let go
    private void captureHotkey() {
        final Dialog d = new Dialog(activity);
        d.requestWindowFeature(Window.FEATURE_NO_TITLE);
        LinearLayout box = new LinearLayout(activity);
        box.setOrientation(LinearLayout.VERTICAL);
        box.setPadding(dp(16), dp(12), dp(16), dp(14));
        box.setBackground(plate(PANEL, 2, 0));
        TextView title = text(tr("MOD UI hotkey", "MOD 界面快捷键"), 20, GOLD, true);
        title.setTypeface(Typeface.create(Typeface.SERIF, Typeface.BOLD));
        box.addView(title);
        box.addView(rule(RED, 2));
        box.addView(rule(GREEN, 1));
        TextView ask = text(tr("Hold up to three controller buttons together, then let go.", "在手柄上同时按住最多三个按键，然后松开。"),
                16, CREAM, false);
        ask.setPadding(0, dp(12), 0, dp(4));
        box.addView(ask);
        TextView now = text(tr("Now: ", "当前：") + hotkeyName(ModUi.hotkey(ini)), 14, DIM, false);
        box.addView(now);

        final java.util.List<Integer> chord = new java.util.ArrayList<>();
        final java.util.Set<Integer> down = new java.util.HashSet<>();
        final boolean[] done = {false};
        final java.util.function.Consumer<String> take = value -> {
            if (done[0]) return;
            done[0] = true;
            ini.set(ModUi.HOTKEY, value);
            d.dismiss();
            changed();
            build();
        };
        d.setOnKeyListener((dialog, keyCode, ev) -> {
            boolean pad = (ev.getSource() & (InputDevice.SOURCE_GAMEPAD | InputDevice.SOURCE_JOYSTICK
                    | InputDevice.SOURCE_DPAD | InputDevice.SOURCE_KEYBOARD)) != 0 && ev.getDevice() != null
                    && !ev.getDevice().isVirtual();
            if (keyCode == KeyEvent.KEYCODE_VOLUME_UP || keyCode == KeyEvent.KEYCODE_VOLUME_DOWN) return false;
            if (!pad) return keyCode != KeyEvent.KEYCODE_BACK;
            if (ev.getAction() == KeyEvent.ACTION_DOWN && ev.getRepeatCount() == 0) {
                down.add(keyCode);
                if (!chord.contains(keyCode) && chord.size() < 3) chord.add(keyCode);
                int[] keys = new int[chord.size()];
                for (int i = 0; i < keys.length; i++) keys[i] = chord.get(i);
                now.setText(hotkeyName(keys));
            } else if (ev.getAction() == KeyEvent.ACTION_UP) {
                down.remove(keyCode);
                if (down.isEmpty() && !chord.isEmpty()) {
                    StringBuilder v = new StringBuilder();
                    for (int k : chord) v.append(v.length() > 0 ? "+" : "").append(k);
                    take.accept(v.toString());
                }
            }
            return true;
        });

        LinearLayout buttons = new LinearLayout(activity);
        buttons.setPadding(0, dp(14), 0, 0);
        TextView usual = button(tr("Default", "默认"), false);
        usual.setOnClickListener(v -> take.accept(null));
        TextView none = button(tr("None", "无"), false);
        none.setOnClickListener(v -> take.accept("0"));
        TextView cancel = button(tr("Cancel", "取消"), true);
        cancel.setOnClickListener(v -> {
            done[0] = true;
            d.dismiss();
        });
        LinearLayout.LayoutParams third = new LinearLayout.LayoutParams(0, dp(46), 1);
        third.setMargins(dp(4), 0, dp(4), 0);
        buttons.addView(usual, third);
        buttons.addView(none, new LinearLayout.LayoutParams(third));
        buttons.addView(cancel, new LinearLayout.LayoutParams(third));
        box.addView(buttons);

        d.setContentView(box);
        Window w = d.getWindow();
        if (w != null) {
            w.setBackgroundDrawable(new ColorDrawable(Color.TRANSPARENT));
            DisplayMetrics m = activity.getResources().getDisplayMetrics();
            w.setLayout(Math.min(dp(520), (int) (m.widthPixels * 0.7f)), ViewGroup.LayoutParams.WRAP_CONTENT);
        }
        d.show();
        box.requestFocus();
    }

    // ---- the layout editor's Edit: one control -------------------------------------------------------------------------

    // Its size and opacity (sliders), its colour, edge and a picture; the shape too but for the d-pad's (a stick's is
    // its base's), an icon and a label for a button or the menu; and for a button tap to hold, and a combo's buttons
    // or a copy of a console button as a new one (a combo of that button alone, so a controller button can fire it too)
    void editControl(Controls.Ctl c) {
        final int k = c.ordinal();
        final Dialog d = new Dialog(activity);
        d.requestWindowFeature(Window.FEATURE_NO_TITLE);
        LinearLayout box = new LinearLayout(activity);
        box.setOrientation(LinearLayout.VERTICAL);
        box.setPadding(dp(16), dp(12), dp(16), dp(14));
        box.setBackground(plate(PANEL, 2, 0));
        TextView title = text(cfg.name(c), 20, GOLD, true);
        title.setTypeface(Typeface.create(Typeface.SERIF, Typeface.BOLD));
        box.addView(title);
        box.addView(rule(RED, 2));
        box.addView(rule(GREEN, 1));

        final TextView sizeValue = text("", 16, GOLD, false);
        Runnable showSize = () -> sizeValue.setText(Math.round(cfg.place[k][2] * 100) + " %");
        showSize.run();
        box.addView(rowView(tr("Size", "大小"), sizeValue, tr("This control's size, on top of the size of all of them (Touch controls).", "在全局触控大小基础上调整此控件的大小。")));
        SeekBar size = new SeekBar(activity);
        size.setMax(200);  // (50 % to 250 %)
        size.setProgress(Math.round(cfg.place[k][2] * 100) - 50);
        size.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar bar, int value, boolean user) {
                if (!user) return;
                cfg.place[k][2] = (value + 50) / 100f;
                showSize.run();
                changed();
            }
            @Override public void onStartTrackingTouch(SeekBar bar) {}
            @Override public void onStopTrackingTouch(SeekBar bar) {}
        });
        size.setPadding(dp(16), dp(10), dp(16), dp(10));
        goldBar(size);
        box.addView(size);

        final TextView alphaValue = text("", 16, GOLD, false);
        Runnable showAlpha = () -> alphaValue.setText(cfg.alpha[k] + " %");
        showAlpha.run();
        box.addView(rowView(tr("Opacity", "不透明度"), alphaValue, tr("This control's opacity, as a share of the opacity of all of them.", "相对于全局不透明度调整此控件的不透明度。")));
        SeekBar alpha = new SeekBar(activity);
        alpha.setMax(90);  // (10 % to 100 %)
        alpha.setProgress(cfg.alpha[k] - 10);
        alpha.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar bar, int value, boolean user) {
                if (!user) return;
                cfg.alpha[k] = value + 10;
                showAlpha.run();
                changed();
            }
            @Override public void onStartTrackingTouch(SeekBar bar) {}
            @Override public void onStopTrackingTouch(SeekBar bar) {}
        });
        alpha.setPadding(dp(16), dp(10), dp(16), dp(10));
        goldBar(alpha);
        box.addView(alpha);

        final boolean button = Controls.isButton(c), stick = Controls.isStick(c),
                menu = c == Controls.Ctl.MENU || c == Controls.Ctl.MOD;
        if (c != Controls.Ctl.DPAD)
            box.addView(pickRow(tr("Shape", "形状"), chinese ? new String[]{"默认", "圆形", "圆角方形", "方形"} : Controls.SHAPES, () -> cfg.shape[k], v -> cfg.shape[k] = v,
                    stick ? tr("The stick's base. Its own: the usual circle.", "摇杆底座的形状。默认使用圆形。")
                          : tr("Its own: the usual look (a combo of one button looks like that button).", "默认使用原有外观，单键组合与对应的按键外观一致。")));
        if (button || menu)
            box.addView(pickRow(tr("Icon", "图标"), chinese ? new String[]{"无（文字）", "上箭头", "下箭头", "左箭头", "右箭头", "星形", "菱形", "盾牌", "刀刃", "准星", "短横线"} : Controls.ICONS, () -> cfg.icon[k], v -> cfg.icon[k] = v,
                    menu ? (c == Controls.Ctl.MOD ? tr("An icon instead of MOD.", "用图标替换 MOD 文字。") : tr("An icon instead of the gear.", "用图标替换齿轮。")) : tr("An icon instead of the button's letters.", "用图标替换按键文字。")));
        final TextView edgeValue = text("", 16, GOLD, false);
        Runnable showEdge = () -> edgeValue.setText(cfg.outline[k] == 0 ? tr("None", "无") : cfg.outline[k] + " %");
        showEdge.run();
        box.addView(rowView(tr("Outline", "描边"), edgeValue, tr("The width of its bronze edge (0: none).", "铜色边框宽度，0 表示无边框。")));
        SeekBar edge = new SeekBar(activity);
        edge.setMax(300);
        edge.setProgress(cfg.outline[k]);
        edge.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar bar, int value, boolean user) {
                if (!user) return;
                cfg.outline[k] = value;
                showEdge.run();
                changed();
            }
            @Override public void onStartTrackingTouch(SeekBar bar) {}
            @Override public void onStopTrackingTouch(SeekBar bar) {}
        });
        edge.setPadding(dp(16), dp(10), dp(16), dp(10));
        goldBar(edge);
        box.addView(edge);
        box.addView(pickRow(tr("Colour", "颜色"), chinese ? new String[]{"酒红（游戏默认）", "黑色", "深红", "藏蓝", "墨绿", "紫色", "灰蓝", "透明"} : Controls.COLORS, () -> cfg.color[k], v -> cfg.color[k] = v,
                stick ? tr("The stick's base and knob. Clear: only the edges, the game showing through.", "摇杆底座与摇杆帽的颜色。透明时仅显示轮廓。")
                      : tr("Its plate. Clear: only its edge, the game showing through.", "按键底色。透明时仅显示轮廓。")));
        final TextView labelValue = text("", 16, GOLD, false);
        Runnable showLabel = () -> labelValue.setText(cfg.label[k] == null ? tr("Its own", "默认") : cfg.label[k]);
        showLabel.run();
        LinearLayout labelRow = rowView(tr("Label", "标签"), labelValue, tr("Your own text on it instead of ", "输入自定义文字以替换") + (menu ? tr("the gear", "齿轮图标") : tr("its letters", "按键文字"))
                + tr(" (empty: its own).", "（留空使用默认值）。"));
        labelRow.setOnClickListener(v -> new TextPrompt(activity, chinese).show(tr("Label", "标签"), tr("Empty for its own", "留空使用默认值"),
                cfg.label[k] == null ? "" : cfg.label[k], "", 0, 12, false, false, text -> {
                    if (text == null) return;
                    cfg.label[k] = text.trim().isEmpty() ? null : text.trim();
                    showLabel.run();
                    changed();
                }));
        if (button || menu) box.addView(labelRow);
        final TextView imageValue = text(cfg.image[k] ? tr("Yours  ▾", "自定义  ▾") : tr("None  ▾", "无  ▾"), 16, GOLD, false);
        LinearLayout imageRow = rowView(tr("Picture", "图像"), imageValue,
                (stick ? tr("An image of yours as the stick's knob (it moves with your thumb)", "使用自选图片作为随手指移动的摇杆帽")
                       : tr("An image of yours instead of its drawing", "使用自选图片替换控件图案"))
                        + tr(", kept with the layout (and in exported layouts). Pressed, it darkens.", "，图片随布局保存和导出。按下时图片会变暗。"));
        imageRow.setOnClickListener(v -> dropDown(tr("Picture", "图像"), cfg.image[k] ? new String[] {tr("Choose another…", "更换图片…"), tr("Remove", "移除")}
                                                                            : new String[] {tr("Choose an image…", "选择图片…")}, -1, i -> {
            if (i == 1) {
                cfg.clearImage(c);
                imageValue.setText(tr("None  ▾", "无  ▾"));
                changed();
                return;
            }
            d.dismiss();
            activity.pickImage(c, () -> editControl(c));  // (the dialog again, with the picture)
        }));
        box.addView(imageRow);
        box.addView(pickRow(tr("Picture shape", "图片形状"), new String[] {stick ? tr("Cut round", "裁成圆形") : tr("Cut to the button", "裁成按键形状"), tr("The picture's own", "原图比例")},
                () -> cfg.fit[k] ? 1 : 0, v -> cfg.fit[k] = v != 0,
                stick ? tr("Cut round: the picture fills the knob, with its edge. The picture's own: only the picture, "
                        + "in its proportions (the stick still takes touches on its whole base).", "裁成圆形会填满摇杆帽并保留边框。原图比例仅显示图片，整个底座仍可触控。")
                      : tr("Cut to the control: the picture fills its shape, with its edge. The picture's own: only the "
                        + "picture, in its proportions, and only where it's drawn takes touches (a see-through "
                        + "background doesn't).", "裁成按键形状会填满控件并保留边框。按原图比例显示时，只有非透明部分响应触控。")));
        if (button) {
            box.addView(pickRow(tr("Tap to hold", "点按锁定"), new String[] {tr("Off", "关"), tr("On", "开")}, () -> cfg.hold[k] ? 1 : 0, v -> cfg.hold[k] = v != 0,
                    tr("On: a tap presses it and keeps it pressed until the next tap (a small diamond shows it).", "点按一次保持按下，再点一次松开。小菱形表示已锁定。")));
            if (c.combo()) {
                LinearLayout r = rowView(tr("Buttons and controller button", "按键组合及触发键"), text("›", 16, GOLD, false),
                        tr("What it presses, and the controller button that does the same.", "设置要同时按下的按键，以及触发此组合的实体按键。"));
                r.setOnClickListener(v -> {
                    d.dismiss();
                    editCombo(c.comboIndex());
                });
                box.addView(r);
            } else {
                LinearLayout r = rowView(tr("Copy as a new button", "复制为新按键"), text("›", 16, GOLD, false),
                        tr("Adds another ", "添加一个新的 ") + Controls.label(c) + tr(" button (a combo of it alone: Settings → Combos, where a "
                                + "controller button can fire it too).", "按键，可在「设置 → 组合键」中为其绑定实体按键。"));
                r.setOnClickListener(v -> {
                    int j = cfg.freeCombo();
                    if (j < 0) return;
                    Controls.Ctl copy = Controls.Ctl.values()[Controls.Ctl.COMBO1.ordinal() + j];
                    cfg.combo[j] = cfg.buttons(c);
                    cfg.comboPad[j] = 0;
                    float[] p = cfg.place[copy.ordinal()];
                    p[0] = Math.min(0.95f, cfg.place[k][0] + 0.06f);
                    p[1] = cfg.place[k][1];
                    p[2] = cfg.place[k][2];
                    cfg.shown[copy.ordinal()] = true;
                    cfg.shape[copy.ordinal()] = cfg.shape[k];
                    cfg.icon[copy.ordinal()] = cfg.icon[k];
                    cfg.hold[copy.ordinal()] = cfg.hold[k];
                    cfg.color[copy.ordinal()] = cfg.color[k];
                    cfg.alpha[copy.ordinal()] = cfg.alpha[k];
                    cfg.label[copy.ordinal()] = cfg.label[k];
                    cfg.outline[copy.ordinal()] = cfg.outline[k];
                    d.dismiss();
                    changed();
                });
                box.addView(r);
            }
        }

        TextView ok = button(tr("Done", "完成"), true);
        ok.setOnClickListener(v -> d.dismiss());
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(46));
        lp.setMargins(dp(4), dp(10), dp(4), 0);
        box.addView(ok, lp);
        ScrollView sv = new ScrollView(activity);
        sv.addView(box);
        d.setContentView(sv);
        Window w = d.getWindow();
        if (w != null) {
            w.setBackgroundDrawable(new ColorDrawable(Color.TRANSPARENT));
            DisplayMetrics m = activity.getResources().getDisplayMetrics();
            w.setLayout(Math.min(dp(560), (int) (m.widthPixels * 0.8f)), ViewGroup.LayoutParams.WRAP_CONTENT);
        }
        d.show();
    }

    private void goldBar(SeekBar bar) {
        bar.setProgressTintList(ColorStateList.valueOf(GOLD));
        bar.setProgressBackgroundTintList(ColorStateList.valueOf(EDGE));
        bar.setThumbTintList(ColorStateList.valueOf(GOLD));
    }

    // A row in a dialog: its value, and a tap opens the drop-down (two options: flips)
    private LinearLayout pickRow(String title, String[] options, IntSupplier get, IntConsumer set, String help) {
        final TextView value = text("", 16, GOLD, false);
        final boolean many = options.length > 2;
        Runnable show = () -> value.setText(options[get.getAsInt()] + (many ? "  ▾" : ""));
        show.run();
        LinearLayout r = rowView(title, value, help);
        r.setOnClickListener(v -> {
            if (!many) {
                set.accept(1 - get.getAsInt());
                show.run();
                changed();
                return;
            }
            dropDown(title, options, get.getAsInt(), i -> {
                set.accept(i);
                show.run();
                changed();
            });
        });
        return r;
    }

    // ---- combo editor ---------------------------------------------------------------------------------------------------

    void editCombo(int k) {
        final Dialog d = new Dialog(activity);
        d.requestWindowFeature(Window.FEATURE_NO_TITLE);
        LinearLayout box = new LinearLayout(activity);
        box.setOrientation(LinearLayout.VERTICAL);
        box.setPadding(dp(16), dp(12), dp(16), dp(14));
        box.setBackground(plate(PANEL, 2, 0));
        TextView title = text(cfg.comboName(k), 20, GOLD, true);
        title.setTypeface(Typeface.create(Typeface.SERIF, Typeface.BOLD));
        box.addView(title);
        box.addView(rule(RED, 2));
        box.addView(rule(GREEN, 1));
        box.addView(text(tr("The buttons it presses:", "同时按下的按键"), 14, DIM, false));

        // buttons in rows of eight: tap one to add or remove it
        final long[] mask = {cfg.combo[k]};
        final long[] padMask = {cfg.comboPad[k]};
        LinearLayout grid = new LinearLayout(activity);
        grid.setOrientation(LinearLayout.VERTICAL);
        LinearLayout line = null;
        final TextView[] chips = new TextView[Controls.BUTTONS.length];
        for (int i = 0; i < Controls.BUTTONS.length; i++) {
            if (i % 8 == 0) {
                line = new LinearLayout(activity);
                grid.addView(line);
            }
            final int b = i;
            TextView chip = text(Controls.NAMES[i], 16, CREAM, true);
            chip.setGravity(Gravity.CENTER);
            chip.setOnClickListener(v -> {
                mask[0] ^= Controls.BUTTONS[b];
                paintChip(chips[b], (mask[0] & Controls.BUTTONS[b]) != 0);
            });
            chips[i] = chip;
            paintChip(chip, (mask[0] & Controls.BUTTONS[i]) != 0);
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(0, dp(44), 1);
            lp.setMargins(dp(3), dp(3), dp(3), dp(3));
            line.addView(chip, lp);
        }
        box.addView(grid);

        // the controller button that fires it: one of Controls.PAD_TRIGGERS, picked from a drop-down
        final TextView padValue = text("", 16, GOLD, false);
        LinearLayout padRow = rowView(tr("Controller button", "手柄触发键"), padValue,
                tr("While it is held, the game gets the combo's buttons instead of it.", "按住此键时，游戏收到组合中设置的按键。"));
        Runnable showPad = () -> padValue.setText((padMask[0] == 0 ? tr("None", "无") : Controls.names(padMask[0])) + "  ▾");
        showPad.run();
        padRow.setOnClickListener(v -> {
            int at = 0;
            final String[] names = new String[Controls.PAD_TRIGGERS.length];
            for (int i = 0; i < names.length; i++) {
                if (Controls.PAD_TRIGGERS[i] == padMask[0]) at = i;
                names[i] = Controls.PAD_TRIGGERS[i] == 0 ? tr("None", "无") : Controls.names(Controls.PAD_TRIGGERS[i]);
            }
            dropDown(tr("Controller button", "手柄触发键"), names, at, i -> {
                padMask[0] = Controls.PAD_TRIGGERS[i];
                showPad.run();
            });
        });
        box.addView(padRow);

        LinearLayout buttons = new LinearLayout(activity);
        buttons.setPadding(0, dp(10), 0, 0);
        TextView clear = button(tr("Clear", "清除"), false);
        clear.setOnClickListener(v -> {
            cfg.combo[k] = 0;
            cfg.comboPad[k] = 0;
            d.dismiss();
            changed();
            build();
        });
        TextView ok = button(tr("Save", "保存"), true);
        ok.setOnClickListener(v -> {
            cfg.combo[k] = mask[0];
            cfg.comboPad[k] = mask[0] == 0 ? 0 : padMask[0];
            d.dismiss();
            changed();
            build();
        });
        LinearLayout.LayoutParams half = new LinearLayout.LayoutParams(0, dp(46), 1);
        half.setMargins(dp(4), 0, dp(4), 0);
        buttons.addView(clear, half);
        buttons.addView(ok, new LinearLayout.LayoutParams(half));
        box.addView(buttons);

        d.setContentView(box);
        Window w = d.getWindow();
        if (w != null) {
            w.setBackgroundDrawable(new ColorDrawable(Color.TRANSPARENT));
            DisplayMetrics m = activity.getResources().getDisplayMetrics();
            w.setLayout(Math.min(dp(620), (int) (m.widthPixels * 0.85f)), ViewGroup.LayoutParams.WRAP_CONTENT);
        }
        d.show();
    }

    private void paintChip(TextView chip, boolean on) {
        chip.setTextColor(on ? 0xFF281010 : CREAM);
        chip.setBackground(plate(on ? GOLD : ROW, 1, 4));
    }

    // ---- mods and cheats (ModFiles; the runtime's side is fs.cpp and cheats.cpp) ----------------------------------------

    // What the last export or convert did, shown at the top of Saves (GameActivity sets it)
    static String saveMessage;

    private void buildSaves() {
        if (folder == null) {
            note(tr("Pick the game folder first.", "请先选择游戏目录。"));
            return;
        }
        if (saveMessage != null) note(saveMessage);
        header(tr("This save", "当前存档"));
        note(SaveFiles.describe(folder, chinese) + tr(" The port uses the Switch's save files, so a save exported from here works on the "
                + "Switch (with JKSV or Checkpoint), Eden and Yuzu as it is.", "本移植版使用 Switch 存档格式，导出的存档可直接用于 Switch（JKSV 或 Checkpoint）、Eden 和 Yuzu。"));
        action(tr("Export (Switch, Eden)", "导出（Switch、Eden）"), tr("Saves a zip of the save folder: data001Slot.bin, data00-1.bin.", "将含有 data001Slot.bin、data00-1.bin 的存档目录打包为 ZIP。"), () -> {
            dialog.dismiss();
            activity.exportSave(false);
        });
        action(tr("Export for Steam", "导出至 Steam"), tr("Converts the save to the PC version's and zips it in win64_save/, to put in Steam's "
                + "...\\1446780\\remote\\win64_save (close the game first, and turn off Steam Cloud for it or it may bring the "
                + "old one back). It's encrypted for the Steam account below.", "转换为 PC 版格式，并按下方 Steam 账户加密。解压到 Steam 的 1446780/remote/win64_save 目录。请先关闭游戏及 Steam 云同步，以免旧存档被自动恢复。"), () -> {
            dialog.dismiss();
            activity.exportSave(true);
        });

        header(tr("Steam account", "Steam 账户"));
        String account = SaveFiles.steamAccount(folder);
        note(account != null ? tr("Saves for Steam are encrypted for ", "Steam 存档将为以下账户加密：") + account + "." : tr("None yet: a save for Steam comes "
                + "out unencrypted. The PC version encrypts its saves with a key that comes from the Steam ID, and the only way "
                + "to get that key is from a save the game encrypted for that account, so the port learns it from any Steam "
                + "save it opens. Played in the port first? Start the PC game once on that account, make a character and "
                + "save, then pick that save here.", "尚未识别账户，导出的 Steam 存档不会加密。请用该账户在 PC 版创建角色并保存，然后在此选择该存档以读取加密信息。"));
        action(tr("Set from a Steam save", "从 Steam 存档识别账户"), tr("Pick a save from that Steam account (a zip or its win64_save folder). Nothing is "
                + "imported: the port only learns the account's key and the PC version's layout from it.", "选择该账户的存档 ZIP 或 win64_save 目录。仅识别账户密钥和存档结构，不导入存档。"), () -> dropDown(
                tr("Set from a Steam save", "从 Steam 存档识别账户"), new String[] {tr("From a .zip", "从 ZIP 文件"), tr("From a folder", "从文件夹")}, -1, i -> {
                    dialog.dismiss();
                    activity.pickSave(i == 1, 2);
                }));

        header(tr("Import", "导入"));
        note(tr("Takes the place of this save. The one here is backed up first. A Steam save is converted on the way in: it's "
                + "decrypted with your Steam ID (read from the save itself when it can be, asked for when not), then filled "
                + "into this save's layout. The game closes right after, so it can't write the old save back.", "先备份当前存档，再替换。Steam 存档会先用 Steam ID 解密并转换，需要时会提示输入 ID。导入完成后游戏会关闭，避免旧存档被写回。"));
        String[] from = {tr("From a .zip", "从 ZIP 文件"), tr("From a folder", "从文件夹")};
        action(tr("Import a save", "导入存档"), tr("A zip or a folder with data00-1.bin and a data00#Slot.bin, from the Switch, Eden, Yuzu, "
                + "another copy of the port, or Steam (win64_save).", "选择含 data00-1.bin 和 data00#Slot.bin 的存档 ZIP 或目录。支持 Switch、Eden、Yuzu、其他移植版及 Steam 的 win64_save。"), () -> dropDown(tr("Import a save", "导入存档"), from, -1, i -> {
                    dialog.dismiss();
                    activity.pickSave(i == 1, 0);
                }));
        List<File> backups = SaveFiles.backups(folder);
        if (!backups.isEmpty()) {
            String[] names = new String[backups.size()];
            for (int i = 0; i < names.length; i++) names[i] = backups.get(i).getName();
            action(tr("Restore a backup", "恢复备份"), tr("The save as it was before an import or restore (the last 10 are kept).", "恢复到导入或还原之前的状态，保留最近 10 个备份。"),
                    () -> dropDown(tr("Restore a backup", "恢复备份"), names, -1, i -> dropDown(tr("Restore the save from ", "从以下备份恢复存档：") + names[i] + "?",
                            new String[] {tr("Keep this one", "保留当前存档"), tr("Restore it", "恢复备份")}, -1, k -> {
                                if (k == 1) {
                                    try {
                                        String done = SaveFiles.restore(folder, backups.get(i));
                                        dialog.dismiss();
                                        activity.saveOver(done);
                                        return;
                                    } catch (IOException e) {
                                        saveMessage = tr("Couldn't restore it: ", "恢复失败：") + e.getMessage();
                                    }
                                }
                                build();
                            })));
        }

        header(tr("Convert", "转换"));
        note(tr("A save from one platform into a zip for the other, without touching this one. Steam to Switch fills this "
                + "save's layout (have one here first for the best result). Switch to Steam fills the Steam account's save "
                + "and is encrypted for it.", "将存档转换成另一平台的 ZIP 文件，保留当前存档。Steam 转 Switch 会沿用现有存档结构，建议先在本机建档。转至 Steam 时将按该账户加密。"));
        action(tr("Convert a save", "转换存档"), tr("Pick a Steam save to get the Switch's, or a Switch save to get Steam's; then where to "
                + "save the zip.", "选择 Steam 或 Switch 存档以转换到另一平台，然后选择 ZIP 的保存位置。"), () -> dropDown(tr("Convert a save", "转换存档"), from, -1, i -> {
                    dialog.dismiss();
                    activity.pickSave(i == 1, 1);
                }));
    }

    private void buildMods() {
        header(tr("Mods", "MOD"));
        if (GameActivity.luaBuild(activity)) {
            action(tr("MOD UI", "MOD 界面"), tr("Open script controls.", "打开 MOD 脚本的控制界面。"), () -> {
                dialog.dismiss();
                activity.showModUi();
            });
            if (folder != null) {
                int[] keys = ModUi.hotkey(ini);
                action(tr("MOD UI controller hotkey: ", "MOD 界面手柄快捷键：") + hotkeyName(keys),
                        tr("Buttons pressed together on the controller open and close the MOD UI. L3 + R3 by default, as in "
                                + "REFramework.", "在手柄上同时按下这些按键可打开或关闭 MOD 界面。默认为 L3 + R3，与 REFramework 相同。"),
                        this::captureHotkey);
                number(tr("MOD UI opacity", "MOD 界面不透明度"),
                        tr("How much of the game the MOD UI's window hides. Lower to watch the hunt while changing values.",
                                "MOD 界面窗口遮挡游戏画面的程度。调低后可以一边调整数值一边观察狩猎。"),
                        30, 100, 5, " %", () -> ModUi.opacity(ini), v -> ini.set(ModUi.OPACITY, Integer.toString(v)));
            }
        }
        if (folder == null) {
            note(tr("Pick the game folder first.", "请先选择游戏目录。"));
            return;
        }
        List<ModFiles.Mod> mods = ModFiles.list(folder);
        if (activity.restartPending)
            action(tr("Restart to apply your changes", "重启以应用更改"),
                    tr("Mods with game files were added, removed, or turned on or off: those load when the game starts.",
                            "已添加、移除或启停含游戏资源的 MOD，这些资源在游戏启动时加载。"), this::askRestart);
        note(tr("Each mod is a folder in ", "每个 MOD 的目录位于 ") + ModFiles.modsDir(folder) + tr(", laid out as for Eden: romfs/ with the game files it "
                + "replaces, cheats/ with its cheat files. Scripts and cheats apply right away; game files the next time the "
                + "game starts.", "。目录结构与 Eden 相同：romfs/ 放替换的游戏资源，cheats/ 放金手指。脚本和金手指立即生效，游戏资源需重启游戏。"));
        action(tr("Add a mod archive", "添加 MOD 压缩包"),
                tr("ZIP, 7z, RAR or TAR. Import game resources, REFramework scripts and their data. Restart to apply.", "支持 ZIP、7z、RAR 和 TAR。可导入游戏资源、REFramework 脚本及数据。重启游戏生效。"),
                () -> {
                    dialog.dismiss();
                    activity.importMod();
                });
        for (ModFiles.Mod m : mods) {
            java.util.List<String> contents = new java.util.ArrayList<>();
            if (m.romfs) contents.add(tr("Game files", "游戏资源"));
            if (m.cheats) contents.add(tr("Cheats", "金手指"));
            if (new File(m.dir, "reframework").isDirectory()) contents.add(tr("REFramework scripts/data", "REFramework 脚本及数据"));
            String what = contents.isEmpty() ? tr("No supported files", "没有支持的文件") : android.text.TextUtils.join(", ", contents);
            toggle(m.name, what + tr(". In ", "。目录：") + m.dir.getName() + ".", () -> m.on ? 1 : 0, v -> {
                if (!ModFiles.setOn(m, v != 0)) android.widget.Toast.makeText(activity, tr("Couldn't rename ", "无法重命名 ") + m.dir.getName(),
                        android.widget.Toast.LENGTH_LONG).show();
                else modChanged(m.dir);
                build();
            });
        }
        if (!mods.isEmpty()) {
            String[] names = new String[mods.size()];
            for (int i = 0; i < names.length; i++) names[i] = mods.get(i).name;
            action(tr("Remove a mod", "移除 MOD"), tr("Deletes the mod's folder and everything in it.", "删除该 MOD 的目录及其中所有文件。"), () -> dropDown(tr("Remove a mod", "移除 MOD"), names, -1,
                    i -> dropDown(tr("Delete ", "删除 ") + names[i] + "?", new String[] {tr("Keep it", "保留"), tr("Delete it", "删除")}, -1, k -> {
                        if (k == 1) {
                            boolean game = ModFiles.gameFiles(mods.get(i).dir), lua = ModFiles.scripts(mods.get(i).dir);
                            if (ModFiles.delete(mods.get(i))) modChanged(game, lua);
                        }
                        build();
                    })));
        }

        String buildId = ModFiles.buildId(folder);
        header(tr("Cheats", "金手指"));
        note(buildId != null
                ? tr("Atmosphere cheats, as in Eden. This game's build ID is ", "使用与 Eden 兼容的 Atmosphère 金手指。游戏的构建 ID 为 ") + buildId + tr(", so a mod's cheat file is cheats/", "，对应金手指文件为 cheats/")
                        + buildId + tr(".txt. A cheat turned on starts working within a second; the master code of its file "
                        + "({...}) runs along with it.", ".txt。开启后约一秒生效，文件中的主代码（{...}）会同时运行。")
                : tr("Can't read the game's build ID (the exefs isn't where the setup found it), so no cheats can be listed.", "无法读取游戏构建 ID。请确认 exefs 路径与初始设置一致。"));
        action(tr("Add a cheat file (.txt)", "添加金手指文件（.txt）"),
                tr("Pick a cheat .txt for this version of the game. It goes into a mod of its own, named after the file. A file "
                        + "named after another build ID is for another game version and isn't taken.", "选择与当前游戏版本匹配的金手指文本文件，将以文件名创建独立 MOD。其他构建 ID 的文件不会被导入。"),
                () -> {
                    dialog.dismiss();
                    activity.importCheats();
                });
        boolean any = false;
        for (ModFiles.Mod m : mods) {
            if (!m.on) continue;
            List<File> files = ModFiles.cheatFiles(m, buildId);
            for (File f : files) {
                List<ModFiles.Cheat> cheats = ModFiles.cheats(m, f);
                note("◆ " + m.name + (files.size() > 1 ? " · " + f.getName() : ""));
                for (ModFiles.Cheat c : cheats) {
                    if (c.master) continue;
                    any = true;
                    flag(c.name, tr("From ", "来源：") + m.name + tr(". On: it runs 12 times a second, as on the console.", "。开启后每秒执行 12 次，与主机一致。"), c.key, false);
                }
            }
            int other = ModFiles.otherBuilds(m, buildId);
            if (other > 0) note(m.name + ": " + other + tr(" cheat file(s) for another game version, not used.", "个其他游戏版本的金手指文件已跳过。"));
        }
        if (!any) note(tr("No cheats yet. Add a cheat file, or a mod with a cheats folder.", "尚无金手指。可添加金手指文件，或包含 cheats 目录的 MOD。"));
    }

    // ---- rows -----------------------------------------------------------------------------------------------------------

    private void header(String title) {
        TextView t = text(title, 18, GOLD, true);
        t.setTypeface(Typeface.create(Typeface.SERIF, Typeface.BOLD));
        t.setPadding(dp(4), dp(16), 0, dp(4));
        list.addView(t);
        View line = rule(EDGE, 1);
        list.addView(line);
    }

    private void note(String s) {
        TextView t = text(s, 13, DIM, false);
        t.setPadding(dp(4), dp(6), dp(4), dp(2));
        list.addView(t);
    }

    // A row: title on the left, value and "?" on the right; a tap calls click(+1)
    private TextView row(String title, String value, String help, IntConsumer click, boolean steppers) {
        TextView v = text(value, 16, GOLD, false);
        LinearLayout r = rowView(title, v, help);
        r.setOnClickListener(x -> click.accept(1));
        list.addView(r);
        return v;
    }

    private LinearLayout rowView(String title, TextView value, String help) {
        LinearLayout outer = new LinearLayout(activity);
        outer.setOrientation(LinearLayout.VERTICAL);
        LinearLayout.LayoutParams olp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
        olp.setMargins(0, dp(4), 0, 0);
        outer.setLayoutParams(olp);
        outer.setBackground(plate(ROW, 1, 2));
        LinearLayout r = new LinearLayout(activity);
        r.setGravity(Gravity.CENTER_VERTICAL);
        r.setPadding(dp(14), dp(10), dp(8), dp(10));
        TextView t = text(title, 16, CREAM, false);
        r.addView(t, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        value.setPadding(dp(8), 0, dp(8), 0);
        r.addView(value);
        final TextView helpText = text(help, 13, DIM, false);
        helpText.setPadding(dp(14), 0, dp(14), dp(10));
        helpText.setVisibility(View.GONE);
        TextView q = text("?", 15, GOLD, true);
        q.setGravity(Gravity.CENTER);
        GradientDrawable ring = new GradientDrawable();
        ring.setShape(GradientDrawable.OVAL);
        ring.setStroke(dp(1), EDGE);
        q.setBackground(ring);
        q.setOnClickListener(x -> helpText.setVisibility(helpText.getVisibility() == View.GONE ? View.VISIBLE : View.GONE));
        r.addView(q, new LinearLayout.LayoutParams(dp(30), dp(30)));
        outer.addView(r);
        outer.addView(helpText);
        return outer;
    }

    private void toggle(String title, String help, IntSupplier get, IntConsumer set) {
        String[] onOff = {tr("Off", "关"), tr("On", "开")};
        choice(title, help, onOff, get, set);
    }

    // Two options (on / off): a tap flips it. More than two: a drop-down with the current one highlighted.
    private void choice(String title, String help, String[] options, IntSupplier get, IntConsumer set) {
        final TextView[] value = new TextView[1];
        final boolean list = options.length > 2;
        value[0] = row(title, options[get.getAsInt()] + (list ? "  ▾" : ""), help, step -> {
            if (!list) {
                set.accept((get.getAsInt() + step) % options.length);
                value[0].setText(options[get.getAsInt()]);
                changed();
                return;
            }
            dropDown(title, options, get.getAsInt(), i -> {
                set.accept(i);
                value[0].setText(options[get.getAsInt()] + "  ▾");
                changed();
            });
        }, false);
    }

    // The drop-down: a panel like the menu itself, with a title, the X and one row per option (current one in gold)
    private void dropDown(String title, String[] options, int current, IntConsumer pick) {
        final Dialog d = new Dialog(activity);
        d.requestWindowFeature(Window.FEATURE_NO_TITLE);
        LinearLayout root = new LinearLayout(activity);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackground(plate(PANEL, 2, 0));
        LinearLayout top = new LinearLayout(activity);
        top.setGravity(Gravity.CENTER_VERTICAL);
        top.setPadding(dp(16), dp(8), dp(8), dp(8));
        TextView t = text("◆  " + title, 18, GOLD, true);
        t.setTypeface(Typeface.create(Typeface.SERIF, Typeface.BOLD));
        TextView close = text("✕", 20, CREAM, false);
        close.setPadding(dp(12), dp(2), dp(12), dp(2));
        close.setBackground(plate(0xFF1A0C09, 1, 2));
        close.setOnClickListener(v -> d.dismiss());
        top.addView(t, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        top.addView(close);
        root.addView(top);
        root.addView(rule(RED, 2));
        root.addView(rule(GREEN, 1));
        LinearLayout items = new LinearLayout(activity);
        items.setOrientation(LinearLayout.VERTICAL);
        items.setPadding(dp(10), dp(6), dp(10), dp(10));
        for (int i = 0; i < options.length; i++) {
            final int k = i;
            final boolean on = i == current;
            LinearLayout r = new LinearLayout(activity);
            r.setGravity(Gravity.CENTER_VERTICAL);
            r.setPadding(dp(14), dp(12), dp(14), dp(12));
            r.setBackground(on ? gold() : plate(ROW, 1, 2));
            TextView name = text((on ? "◆  " : "") + options[i], 16, on ? 0xFF281010 : CREAM, on);
            r.addView(name, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
            if (on) r.addView(text("◆", 14, 0xFF281010, true));
            r.setOnClickListener(v -> {
                d.dismiss();
                if (k != current) pick.accept(k);
            });
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT);
            lp.topMargin = dp(4);
            items.addView(r, lp);
        }
        ScrollView sv = new ScrollView(activity);
        sv.addView(items);
        root.addView(sv);
        d.setContentView(root);
        Window w = d.getWindow();
        if (w != null) {
            w.setBackgroundDrawable(new ColorDrawable(Color.TRANSPARENT));
            DisplayMetrics m = activity.getResources().getDisplayMetrics();
            w.setLayout(Math.min(dp(460), (int) (m.widthPixels * 0.6f)), ViewGroup.LayoutParams.WRAP_CONTENT);
        }
        d.show();
    }

    // Selected drop-down row: gold with a glow, like the game's menus
    private GradientDrawable gold() {
        GradientDrawable g = new GradientDrawable(GradientDrawable.Orientation.LEFT_RIGHT, new int[] {0xFFD4A43C, 0xFF8E5A22});
        g.setStroke(dp(2), 0xFFF0D080);
        g.setCornerRadius(dp(2));
        return g;
    }

    // Saved menu size, in %
    // Menu size on top of the screen's own: the menus were laid out on a phone about 400 dp tall, so a small phone
    // gets them a little smaller and a tablet bigger (the same share of the screen), then Menu size moves that
    static float screenFactor(android.content.Context c) {
        android.util.DisplayMetrics m = c.getResources().getDisplayMetrics();
        float shortDp = Math.min(m.widthPixels, m.heightPixels) / m.density;
        return Math.max(0.85f, Math.min(1.75f, shortDp / 400f));
    }

    static float sizeFactor(android.content.Context c) {
        return menuScale(c) / 100f * screenFactor(c);
    }

    static int menuScale(android.content.Context c) {
        try {
            return c.getSharedPreferences(PREFS, android.content.Context.MODE_PRIVATE).getInt(SCALE_KEY, SCALE_DEFAULT);
        } catch (RuntimeException e) {
            return SCALE_DEFAULT;
        }
    }

    // A number with − and +: tapping the row also steps up (and wraps to the minimum past the maximum)
    private void number(String title, String help, int min, int max, int step, String unit, IntSupplier get, IntConsumer set) {
        TextView value = text(get.getAsInt() + unit, 16, GOLD, false);
        LinearLayout r = rowView(title, value, help);
        LinearLayout line = (LinearLayout) r.getChildAt(0);
        TextView minus = smallButton("−"), plus = smallButton("+");
        line.addView(minus, 1, new LinearLayout.LayoutParams(dp(40), dp(36)));
        line.addView(plus, 3, new LinearLayout.LayoutParams(dp(40), dp(36)));
        IntConsumer by = d -> {
            set.accept(Math.max(min, Math.min(max, get.getAsInt() + d)));
            value.setText(get.getAsInt() + unit);
            changed();
        };
        minus.setOnClickListener(x -> by.accept(-step));
        plus.setOnClickListener(x -> by.accept(step));
        r.setOnClickListener(x -> {
            int next = get.getAsInt() + step;
            set.accept(next > max ? min : next);
            value.setText(get.getAsInt() + unit);
            changed();
        });
        list.addView(r);
    }

    // ---- game switches --------------------------------------------------------------------------------------------------

    static File gameFolder(GameActivity a) {
        try {
            String path = new String(Files.readAllBytes(new File(a.getFilesDir(), "carpeta.txt").toPath())).trim();
            return path.isEmpty() ? null : new File(path);
        } catch (IOException e) {
            return null;
        }
    }

    // ---- bug report -------------------------------------------------------------------------------------------------

    private static String reportStatus;

    private void trigger(String name, String content) {
        BugReport.trigger(folder, name, content);
    }

    private void saveReport() {
        reportStatus = tr("Saving the report…", "正在保存报告…");
        build();
        BugReport.save(folder, chinese, done -> activity.runOnUiThread(() -> {
            reportStatus = done;
            build();
        }));
    }

    // The ASTC cache (runtime: nvn/astc.cpp): in the app's files folder, or the data folder with astc_cache_in_data_folder
    private File[] astcCacheDirs() {
        return new File[] {new File(activity.getFilesDir(), "astc_cache"), folder != null ? new File(folder, "astc_cache") : null};
    }

    private String astcCacheSize() {
        long bytes = 0;
        for (File dir : astcCacheDirs()) {
            File[] files = dir != null ? dir.listFiles() : null;
            if (files != null)
                for (File f : files) bytes += f.length();
        }
        return String.format(java.util.Locale.ROOT, "%.0f MB", bytes / 1048576.0);
    }

    private void clearAstcCache() {
        for (File dir : astcCacheDirs()) {
            File[] files = dir != null ? dir.listFiles() : null;
            if (files != null)
                for (File f : files) f.delete();
        }
    }

    private GpuInfo gpu;  // the options that apply to this GPU only (all of them before the game's first start)
    private boolean resetArmed;  // Restore default settings asks for a second tap

    // Every switch the menu has ever written as its own file, so Restore default settings clears old folders too.
    private static final String[] SWITCHES = {
        "60fps", "adreno_turbo", "cutout", "system_bars", "cool_down", "no_hud_heat", "no_tight_barriers", "tight_barriers", "int_attribs", "gpu_unswizzle", "async_present", "pipeline_threads_1", "pipeline_threads_3", "pipeline_threads_4", "fg_target_60", "fg_target_90", "fg_target_120", "dynamic_state", "vertex_input_state",
        "no_dynamic_state", "astc_cache_in_data_folder", "bc_astc", "aspect_16_10", "aspect_21_9", "aspect_4_3", "aspect_stretch", "bindless_arrays", "breadcrumbs",
        "capture_button", "count_check", "depth_gequal", "dock_mode", "effects_lower", "effects_lowest", "fast_depth", "fps_20", "fps_25", "fps_40",
        "fps_45", "gpu_occlusion", "hud_battery", "hud_cpu", "hud_frametime", "hud_gpu", "hud_ram", "hud_shaders", "no_hud_battery", "no_hud_cpu",
        "no_hud_frametime", "no_hud_gpu", "no_hud_ram", "no_hud_shaders", "lsfg_no_guard", "keep_dead_functions", "log_lines", "low_memory", "model_detail_high", "model_detail_low",
        "model_detail_lowest", "model_detail_min", "no_adpf", "no_af", "no_affinity", "no_ao", "no_async_shaders", "no_bindless_arrays", "no_bloom",
        "no_chromatic_aberration", "no_depth_of_field", "no_dynamic_resolution", "no_dynamic_shadows",
        "no_effect_ssr", "no_film_grain", "no_foliage_sway", "no_fps", "no_fsr", "no_fxaa", "no_god_rays",
        "no_gpu_occlusion", "no_int_attribs", "no_lens_flare", "no_motion_blur", "no_quad_derivatives", "no_readback_copies",
        "no_return_mali", "no_return_qualcomm", "no_robust", "no_count_check", "no_safe_translation", "no_shadows", "no_short_subgroups",
        "no_shuffle_qualcomm", "no_spirv_opt_brx", "no_area", "no_deferred_clear", "no_store_ops", "no_ubwc", "no_warmup", "perf_stats", "resolution_240", "resolution_360",
        "resolution_432", "resolution_480", "resolution_576", "resolution_648", "resolution_720", "shadow_cache",
        "shadow_half", "shadow_quarter", "short_subgroups", "single_thread_pipelines", "small_object_culling",
        "texture_bias_1", "texture_bias_2", "tv_mode", "ultrawide", "verbose_log", "wide_view",
    };

    private boolean has(String name) {
        return name != null && ini.on(name);
    }

    private void put(String name, boolean on) {
        ini.setOn(name, on);
    }

    // An on / off row backed by a switch in mhr.ini. inverted: the switch turns it off (the no_* ones).
    private void flag(String title, String help, String file, boolean inverted) {
        toggle(title, help, () -> has(file) != inverted ? 1 : 0, v -> put(file, (v != 0) != inverted));
    }

    // What the game is using right now (game_settings.txt, written by the game when a value changes). The menu shows it
    // when it opens: change a setting, close the menu and open it again to see it applied.
    private String gameStatus() {
        String shadow = null, model = null, texture = null;
        try {
            for (String line : Files.readAllLines(new File(folder, "game_settings.txt").toPath())) {
                String[] w = line.trim().split(" ");
                if (w[0].equals("shadow") && w.length >= 3)
                    shadow = w[1].equals(w[2]) ? tr("shadow maps ", "阴影贴图 ") + w[1] + tr(" (game's)", "（游戏默认）") : tr("shadow maps ", "阴影贴图 ") + w[1] + tr(" (game's ", "（游戏默认 ") + w[2] + ")";
                else if (w[0].equals("model") && w.length >= 2)
                    model = Math.abs(Float.parseFloat(w[1]) - 1f) < 0.001f ? tr("model detail game's", "模型细节：游戏默认")
                            : String.format(java.util.Locale.ROOT, tr("model detail x%.2f", "模型细节：×%.2f"), Float.parseFloat(w[1]));
                else if (w[0].equals("texture") && w.length >= 3)
                    texture = w[1].equals(w[2]) ? tr("texture streaming game's", "纹理流送：游戏默认")
                            : tr("textures ", "纹理缩小 ") + (Integer.parseInt(w[1]) - Integer.parseInt(w[2])) + tr(" size(s) smaller", "级");
            }
        } catch (IOException | RuntimeException e) {
            return tr("In the game now: nothing yet (the game writes it while running; this game version may not be supported).", "当前还没有游戏图形数据；此游戏版本可能不受支持。");
        }
        List<String> status = new ArrayList<>();
        for (String part : new String[] {shadow, model, texture})
            if (part != null) status.add(part);
        if (status.isEmpty()) return tr("In the game now: nothing yet (the game writes it once it's in town or a hunt).", "当前还没有图形数据；进入村庄或任务后可查看。");
        return tr("In the game now: ", "当前游戏设置：") + String.join(" · ", status)
                + tr(". Reopen the menu after a change to see it.", "。更改后重新打开菜单可查看最新值。");
    }

    // A choice where each option (but the first, the default) is its own switch
    private void exclusive(String title, String help, String[] options, String[] files) {
        choice(title, help, options, () -> {
            for (int i = files.length - 1; i > 0; i--)
                if (has(files[i])) return i;
            return 0;
        }, v -> {
            for (int i = 1; i < files.length; i++) put(files[i], i == v);
        });
    }

    private void action(String title, String help, Runnable run) {
        row(title, "›", help, x -> run.run(), false);
    }

    // ---- pieces ---------------------------------------------------------------------------------------------------------

    private TextView smallButton(String s) {
        TextView b = text(s, 18, CREAM, true);
        b.setGravity(Gravity.CENTER);
        b.setBackground(plate(0xFF1A0C09, 1, 2));
        return b;
    }

    private TextView button(String s, boolean main) {
        TextView b = text(s, 16, main ? 0xFF281010 : CREAM, true);
        b.setGravity(Gravity.CENTER);
        b.setBackground(plate(main ? GOLD : 0xFF3E2616, 1, 4));
        return b;
    }

    private View rule(int color, int heightDp) {
        View v = new View(activity);
        v.setBackgroundColor(color);
        v.setLayoutParams(new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(heightDp)));
        return v;
    }

    private GradientDrawable plate(int fill, int strokeDp, int radiusDp) {
        GradientDrawable d = new GradientDrawable();
        d.setColor(fill);
        d.setStroke(dp(strokeDp), EDGE);
        d.setCornerRadius(dp(radiusDp));
        return d;
    }

    private TextView text(String s, int sp, int color, boolean bold) {
        TextView t = new TextView(activity);
        t.setText(s);
        t.setTextSize(TypedValue.COMPLEX_UNIT_SP, sp * scale);
        t.setTextColor(color);
        if (bold) t.setTypeface(Typeface.DEFAULT_BOLD);
        return t;
    }

    private int dp(int v) {  // (scaled by the menu size; 1 dp rules never vanish)
        float px = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v * scale, activity.getResources().getDisplayMetrics());
        return v > 0 ? Math.max(1, Math.round(px)) : Math.round(px);
    }
}
