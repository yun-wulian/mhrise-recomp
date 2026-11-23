package io.github.iamwilmar.mhriseport;

import io.github.iamwilmar.mhriseport.SetupActivity.MenuLanguage;

import android.app.NativeActivity;
import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.widget.Toast;
import android.graphics.PixelFormat;
import android.os.Bundle;
import android.os.VibrationEffect;
import android.view.Gravity;
import android.os.Vibrator;
import android.view.View;
import android.view.WindowInsets;
import android.view.WindowInsetsController;
import android.view.WindowManager;

import java.io.File;

// El juego: la NativeActivity de siempre (libnier.so, android_main.cpp), con los controles táctiles encima.
// La NativeActivity se queda con la superficie de su ventana y con su entrada, así que una vista suya no se vería: los
// controles van en una ventana propia, un panel transparente pegado a la del juego (TYPE_APPLICATION_PANEL). No toma el
// foco (FLAG_NOT_FOCUSABLE): las teclas y los ejes del mando siguen yendo al loader. Los toques sí son suyos; los que no
// caen en un control se los pasa al loader para el panel del HUD (nativeTouchPoint).
public final class GameActivity extends NativeActivity implements TouchPad.Host {
    static {
        // (la NativeActivity carga libnier.so por su cuenta; esto la registra también para los métodos native de aquí)
        System.loadLibrary("nier");
    }

    static native void nativeTouch(long buttons, int sticksOn, float lx, float ly, float rx, float ry);
    static native void nativeTouchPoint(int action, float x, float y, long timeNs, int viewWidth, int viewHeight);
    static native void nativePad(float deadzone, float trigger, int rightAxes, long[] combos, long[] mapping);
    static native void nativeTextAnswer(int serial, String text);
    static native void nativeTurbo(boolean on);
    // Load after the NativeActivity has configured the game paths and driver environment.
    private static final class RuntimeLifecycle {
        static { System.loadLibrary("nnsdk"); }
        static native void update(boolean resumed, boolean releasingSurface);
    }
    private boolean resumed;

    private Controls controls;
    private TouchPad pad;
    private ModUi modUi;
    void showModUi() {
        if (modUi != null && modUiReady()) modUi.setMenu(true);
    }

    // The MOD UI draws the Lua scripts' windows: with no script running there's nothing to draw, and the button seemed
    // to do nothing. Says why instead.
    // ---- Mods changed while the game runs ----
    // Scripts reload in place: the Lua runtime reads the mods folder again (a request file it looks for each frame).
    // Game files only load when the game starts, so those wait for a restart, offered by the settings menu. Returns
    // whether this change needs one.
    boolean restartPending;
    boolean modsChanged(boolean gameFiles, boolean scripts) {
        if (gameFiles) restartPending = true;
        if (scripts && !restartPending && !reloadScripts()) restartPending = true;
        if (pad != null) pad.setModsAvailable(luaScripts());
        return restartPending;
    }

    private boolean reloadScripts() {
        if (!luaBuild(this) || ModUi.runtime() == 0) return false;  // (no runtime: only a start loads them)
        File dir = new File(getFilesDir(), "reframework");
        try {
            if (!dir.isDirectory() && !dir.mkdirs()) return false;
            File request = new File(dir, "reload.request");
            return request.exists() || request.createNewFile();
        } catch (java.io.IOException e) {
            return false;
        }
    }

    // Closes the game and opens it again (the launcher's activity, in a new task, started as this process ends)
    void restartGame() {
        Intent launch = getPackageManager().getLaunchIntentForPackage(getPackageName());
        if (launch != null && launch.getComponent() != null)
            startActivity(Intent.makeRestartActivityTask(launch.getComponent()));
        finishAffinity();
        android.os.Process.killProcess(android.os.Process.myPid());
    }

    private boolean luaAtLaunch;  // (the runtime only reads the scripts when the game starts)
    private boolean modUiReady() {
        String why;
        boolean chinese = MenuLanguage.usesChinese(this);
        switch (ModUi.runtime()) {
            case 2: return true;
            case 1:
                why = chinese ? "进入村庄或任务后，脚本会开始运行，届时可打开 MOD 界面。"
                        : "Open the MOD UI after entering the village or a quest, when scripts start.";
                break;
            default:
                if (luaAtLaunch) {
                    why = chinese ? "Lua 运行时未启动，请在「设置 → 调试 → Lua / TDB」查看报告。"
                            : "The Lua runtime didn't start. Settings → Debug → Lua / TDB shows why.";
                    break;
                }
                if (luaScripts()) {
                    why = chinese ? "请重新启动游戏以加载刚添加的 Lua MOD。"
                            : "Lua mods added since the game started load the next time it starts: close the game fully "
                            + "and open it again.";
                    break;
                }
                why = chinese ? "请在「MOD 与金手指」中添加 Lua 脚本，或在「调试」中启用对象浏览器，然后重启游戏。"
                        : "No Lua scripts are installed, so the MOD UI has nothing to show. Add a REFramework script mod in "
                        + "Settings → Mods & cheats, or turn on Debug → Object Explorer, and restart the game.";
        }
        Toast.makeText(this, why, Toast.LENGTH_LONG).show();
        return false;
    }

    // The frame rate the game is limited to (Display → FPS limit): the touch controls animate no faster
    private int gameFps() {
        File folder = SettingsMenu.gameFolder(this);
        if (folder == null) return 30;
        Ini ini = new Ini(folder);
        if (ini.on("60fps")) return 60;
        for (int fps : new int[] {45, 40, 25, 20})
            if (ini.on("fps_" + fps)) return fps;
        return 30;
    }
    // The controller hotkey (android_main.cpp, from a thread of its own) and the floating button
    public void toggleModUi() {
        runOnUiThread(() -> {
            if (modUi == null) return;
            if (modUi.menuOpen()) modUi.setMenu(false);
            else if (modUiReady()) modUi.setMenu(true);
        });
    }

    // A build with the Lua runtime (tools/build_apk.sh LUA_LAB=1, the releases): it ships assets/lua_build. The script
    // menus and the mod UI only exist there.
    private static Boolean lua;
    static boolean luaBuild(android.content.Context c) {
        if (lua == null) {
            try (java.io.InputStream in = c.getAssets().open("lua_build")) {
                lua = true;
            } catch (java.io.IOException e) {
                lua = false;
            }
        }
        return lua;
    }
    private boolean shown;
    private Vibrator vibrator;
    private CaptureButton capture;  // the floating capture button, while Debug → Capture button is on
    private WindowManager.LayoutParams captureLp;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        // Expand into the cutout (Settings → Display): the game's window over the strip beside the camera too
        File cutoutFolder = SettingsMenu.gameFolder(this);
        if (cutoutFolder != null && new Ini(cutoutFolder).on("cutout")) {
            WindowManager.LayoutParams attrs = getWindow().getAttributes();
            attrs.layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_ALWAYS;
            getWindow().setAttributes(attrs);
        }
        controls = new Controls(this);
        vibrator = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
        applyPad();
        pad = new TouchPad(this, controls, this);
        pad.setGameFps(gameFps());
        luaAtLaunch = luaScripts();
        pad.setModsAvailable(luaAtLaunch);
        // La ventana del juego tiene que estar puesta para colgarle el panel: después del primer paso por la cola.
        getWindow().getDecorView().post(this::showPad);
        showSaveMessage();
    }

    // Full screen for real: the Fullscreen theme only drops the status bar, so with three-button navigation (or a visible
    // gesture bar) the bottom bar sat over the game. Both bars hidden every time the window gets the focus (start-up and
    // coming back from another app); a swipe in from the edge shows them for a moment. Settings → Display → Show Android's
    // bars (system_bars) keeps them.
    @Override
    public void onWindowFocusChanged(boolean focused) {
        super.onWindowFocusChanged(focused);
        if (focused) hideSystemBars();
    }

    private void hideSystemBars() {
        File folder = SettingsMenu.gameFolder(this);
        if (folder != null && new Ini(folder).on("system_bars")) return;
        getWindow().setDecorFitsSystemWindows(false);
        WindowInsetsController bars = getWindow().getInsetsController();
        if (bars == null) return;
        bars.hide(WindowInsets.Type.systemBars());
        bars.setSystemBarsBehavior(WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE);
    }

    private void showPad() {
        if (shown || isFinishing()) return;
        View decor = getWindow().getDecorView();
        if (decor.getWindowToken() == null) {  // (aún no: otra vuelta)
            decor.postDelayed(this::showPad, 50);
            return;
        }
        WindowManager.LayoutParams lp = new WindowManager.LayoutParams(WindowManager.LayoutParams.MATCH_PARENT,
                WindowManager.LayoutParams.MATCH_PARENT, WindowManager.LayoutParams.TYPE_APPLICATION_PANEL,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE | WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN
                        | WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS | overlayAcceleration(),
                PixelFormat.TRANSLUCENT);
        lp.token = decor.getWindowToken();
        lp.layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_ALWAYS;
        lp.setTitle("controles");
        try {
            getWindowManager().addView(pad, lp);
            shown = true;
            if (luaBuild(this)) {
                modUi = new ModUi(this);
                if (resumed) modUi.onResume();
            }
            showCapture();
        } catch (RuntimeException e) {
            // (sin panel el juego sigue con el mando físico y el panel del HUD, como antes)
            System.err.println("[controles] no se pudo poner la ventana de los controles: " + e);
        }
    }

    // Settings → Display → Draw controls with the GPU (touch_gpu): the touch controls' windows drawn by Android's
    // RenderThread through the GLES driver, or by the CPU. Mali's GLES driver crashed on that thread mid-hunt (a
    // Mali-G615 on a MediaTek, the game itself fine on Vulkan), so it's off by default on MediaTek and Exynos chips. The
    // controls only change when touched, so the CPU has little to do. Read when the windows are made.
    static boolean touchOnGpu(Ini ini) {
        String v = ini.get("touch_gpu");
        if (v != null) return ini.on("touch_gpu");
        return !maliSoc();
    }

    // A MediaTek or Exynos chip, from the SoC names Android gives
    static boolean maliSoc() {
        String soc = android.os.Build.VERSION.SDK_INT >= 31 ? android.os.Build.SOC_MANUFACTURER + " " + android.os.Build.SOC_MODEL : "";
        String all = (soc + " " + android.os.Build.HARDWARE + " " + android.os.Build.BOARD).toLowerCase(java.util.Locale.ROOT);
        return all.contains("mediatek") || all.matches(".*\\bmt\\d{4}.*") || all.contains("exynos") || all.matches(".*\\bs5e\\d.*");
    }

    private int overlayAcceleration() {
        File folder = SettingsMenu.gameFolder(this);
        boolean gpu = folder != null ? touchOnGpu(new Ini(folder)) : !maliSoc();
        return gpu ? WindowManager.LayoutParams.FLAG_HARDWARE_ACCELERATED : 0;
    }

    // GPU boost (Performance): on while the game is in front, if the switch is on and the GPU is an Adreno
    private void applyTurbo(boolean front) {
        File folder = SettingsMenu.gameFolder(this);
        boolean on = front && folder != null && new Ini(folder).on("adreno_turbo") && GpuInfo.read(folder).adreno();
        try {
            nativeTurbo(on);
        } catch (UnsatisfiedLinkError ignored) {
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        resumed = true;
        RuntimeLifecycle.update(true, false);
        if (modUi != null) modUi.onResume();
        applyTurbo(true);
    }

    @Override
    protected void onPause() {
        if (modUi != null) modUi.onPause();
        applyTurbo(false);
        if (pad != null) pad.releaseAll();  // (un dedo que se queda en un botón al irse la app no lo deja pulsado)
        if (controls != null) controls.saveNow();  // (the app may not come back: swiped away from recents, killed for memory)
        resumed = false;
        RuntimeLifecycle.update(false, false);
        super.onPause();
    }

    @Override
    public void surfaceDestroyed(android.view.SurfaceHolder holder) {
        // NativeActivity waits for the frame consumer to release its old swapchain.
        RuntimeLifecycle.update(resumed, true);
        try {
            super.surfaceDestroyed(holder);
        } finally {
            RuntimeLifecycle.update(resumed, false);
        }
    }

    // The capture button goes in its own small window above the controls' one, only as big as the button, so touches
    // anywhere else go on to the controls and the game. It follows the switch: on and off from the menu, live.
    private void showCapture() {
        if (!shown) return;
        File folder = SettingsMenu.gameFolder(this);
        boolean want = folder != null && new Ini(folder).on(CaptureButton.SWITCH);
        if (want == (capture != null)) return;
        if (!want) {
            try {
                getWindowManager().removeViewImmediate(capture);
            } catch (RuntimeException ignored) {
            }
            capture = null;
            return;
        }
        CaptureButton b = new CaptureButton(this, new CaptureButton.Host() {
            @Override public File folder() { return SettingsMenu.gameFolder(GameActivity.this); }
            @Override public void moved(int x, int y) { placeCapture(x, y); }
            @Override public void tick() { GameActivity.this.tick(); }
        });
        WindowManager.LayoutParams lp = new WindowManager.LayoutParams(b.size(), b.size(),
                WindowManager.LayoutParams.TYPE_APPLICATION_PANEL,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE | WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL
                        | WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN | WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS
                        | overlayAcceleration(),
                PixelFormat.TRANSLUCENT);
        lp.token = getWindow().getDecorView().getWindowToken();
        lp.layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_ALWAYS;
        lp.gravity = Gravity.TOP | Gravity.LEFT;
        lp.setTitle("capture");
        try {
            getWindowManager().addView(b, lp);
            capture = b;
            captureLp = lp;
            placeCapture(b.x, b.y);
        } catch (RuntimeException e) {
            System.err.println("[controles] couldn't add the capture button: " + e);
        }
    }

    // Kept on the screen, even if it was left near an edge on a bigger screen or in the other orientation.
    private void placeCapture(int x, int y) {
        if (capture == null) return;
        View decor = getWindow().getDecorView();
        int size = capture.size();
        captureLp.x = capture.x = Math.max(0, Math.min(x, decor.getWidth() - size));
        captureLp.y = capture.y = Math.max(0, Math.min(y, decor.getHeight() - size));
        try {
            getWindowManager().updateViewLayout(capture, captureLp);
        } catch (RuntimeException ignored) {
        }
    }

    @Override
    protected void onDestroy() {
        // NativeActivity will tear down the runtime. Let parked threads finish cleanup with audio paused.
        RuntimeLifecycle.update(false, true);
        if (modUi != null) modUi.dispose();
        if (capture != null) {
            try {
                getWindowManager().removeViewImmediate(capture);
            } catch (RuntimeException ignored) {
            }
            capture = null;
        }
        if (shown) {
            try {
                getWindowManager().removeViewImmediate(pad);
            } catch (RuntimeException ignored) {
            }
            shown = false;
        }
        super.onDestroy();
    }

    // Los ajustes del mando físico, al loader
    void applyPad() {
        try {
            nativePad(controls.padDeadzone / 100f, controls.trigger / 100f, controls.rightAxes, controls.padCombos(),
                    controls.padMapping());
        } catch (UnsatisfiedLinkError e) {
            System.err.println("[controles] el loader no tiene nativePad: " + e);
        }
    }

    void refresh() {
        controls.save();
        applyPad();
        pad.setGameFps(gameFps());
        pad.invalidate();
        showCapture();
        pad.setModsAvailable(luaScripts());
        applyTurbo(true);
    }

    void startEditing() {
        pad.releaseAll();
        pad.editing = true;
        pad.invalidate();
    }

    // ---- TouchPad.Host ----

    @Override
    public void send(long buttons, int sticksOn, float lx, float ly, float rx, float ry) {
        try {
            nativeTouch(buttons, sticksOn, lx, ly, rx, ry);
        } catch (UnsatisfiedLinkError ignored) {
        }
    }

    @Override
    public boolean modTouch(int action, int id, float screenX, float screenY) {
        return modUi != null && modUi.routeTouch(action, id, screenX, screenY);
    }

    @Override
    public void touchPoint(int action, float x, float y, long timeNs, int width, int height) {
        try {
            nativeTouchPoint(action, x, y, timeNs, width, height);
        } catch (UnsatisfiedLinkError ignored) {
        }
    }

    // Un golpe corto del vibrador al pulsar: suave, medio o fuerte. Con un vibrador sin control de fuerza, solo cambia
    // lo que dura.
    @Override
    public void tick() {
        int level = controls.vibration;
        if (level <= 0 || vibrator == null || !vibrator.hasVibrator()) return;
        long ms = level == 1 ? 8 : level == 2 ? 14 : 22;
        int amplitude = level == 1 ? 60 : level == 2 ? 140 : 255;
        vibrator.vibrate(VibrationEffect.createOneShot(ms, vibrator.hasAmplitudeControl() ? amplitude
                                                                                          : VibrationEffect.DEFAULT_AMPLITUDE));
    }

    // Software keyboard (swkbd.cpp, TextPrompt). The loader calls this from its own thread; the answer goes back with the
    // request's serial (null: cancelled). Touch buttons are released first, since the game waits on the text with
    // whatever was held down.
    void askText(int serial, String header, String guide, String initial, String error, int minLength, int maxLength,
                 boolean password, boolean multiline) {
        runOnUiThread(() -> {
            if (pad != null) pad.releaseAll();
            new TextPrompt(this).show(header, guide, initial, error, minLength, maxLength, password, multiline, text -> {
                try {
                    nativeTextAnswer(serial, text);
                } catch (UnsatisfiedLinkError e) {
                    System.err.println("[keyboard] loader has no nativeTextAnswer: " + e);
                }
            });
        });
    }

    @Override
    public void openMenu() {
        pad.releaseAll();
        new SettingsMenu(this, controls).show();
    }

    @Override
    public void editDone() {
        controls.saveNow();
        pad.invalidate();
    }

    @Override
    public void layoutChanged() {
        controls.save();
    }

    @Override
    public void buttonRemoved() {
        controls.save();
        applyPad();  // (its controller buttons go with it)
    }

    // Lua scripts the MOD UI can show: an enabled mod with REFramework scripts, or the Object Explorer. The MOD control
    // is only on the screen then.
    private boolean luaScripts() {
        if (!luaBuild(this)) return false;
        File folder = SettingsMenu.gameFolder(this);
        if (folder == null) return false;
        if (new Ini(folder).on("object_explorer")) return true;
        try {
            for (ModFiles.Mod m : ModFiles.list(folder))
                if (m.on && new File(m.dir, "reframework/autorun").isDirectory()) return true;
        } catch (RuntimeException ignored) {
        }
        return false;
    }

    // ---- pictures for buttons, and layout files (the system's file picker answers in onActivityResult) ----------------

    private static final int PICK_IMAGE = 41, EXPORT_LAYOUT = 42, IMPORT_LAYOUT = 43, IMPORT_MOD = 44, IMPORT_CHEATS = 45, IMPORT_DRIVER = 46, IMPORT_DLL = 47;
    private static final int SAVE_ZIP = 48, SAVE_FOLDER = 49, CONVERT_ZIP = 50, CONVERT_FOLDER = 51, SAVE_OUT = 52, LEARN_ZIP = 53, LEARN_FOLDER = 54;
    private Controls.Ctl imageFor;      // the control a picked image goes to
    private Runnable afterPick;         // the menu or the editor's dialog, shown again

    void pickImage(Controls.Ctl c, Runnable after) {
        imageFor = c;
        afterPick = after;
        Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("image/*");
        start(i, PICK_IMAGE);
    }

    void exportLayout() {
        Intent i = new Intent(Intent.ACTION_CREATE_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("application/zip")
                .putExtra(Intent.EXTRA_TITLE, "mhrise-touch-layout.zip");
        start(i, EXPORT_LAYOUT);
    }

    void importLayout() {
        Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("*/*")
                .putExtra(Intent.EXTRA_MIME_TYPES, new String[] {"application/zip", "application/x-zip-compressed",
                        "application/octet-stream"});
        start(i, IMPORT_LAYOUT);
    }

    // Settings → Mods & cheats: a mod's zip or a cheat file, copied into the mods folder (ModFiles), then the menu again
    void importMod() {
        // SAF providers disagree on MIME types for RAR and 7z. Detect the format from the contents.
        start(new Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("*/*"), IMPORT_MOD);
    }

    void importCheats() {
        start(new Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("*/*")
                .putExtra(Intent.EXTRA_MIME_TYPES, new String[] {"text/plain", "application/octet-stream"}), IMPORT_CHEATS);
    }

    // Settings → GPU driver and Frame generation
    void importDriver() {
        start(new Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("*/*")
                .putExtra(Intent.EXTRA_MIME_TYPES, new String[] {"application/zip", "application/x-zip-compressed",
                        "application/octet-stream"}), IMPORT_DRIVER);
    }

    void importLosslessDll() {
        start(new Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("*/*"), IMPORT_DLL);
    }

    private void importInto(int request, Uri uri) {
        File data = SettingsMenu.gameFolder(this);
        if (uri == null || data == null) {
            new SettingsMenu(this, controls).show();
            return;
        }
        Toast.makeText(this, "Adding it…", Toast.LENGTH_SHORT).show();
        new Thread(() -> {
            String done;
            ModFiles.ImportResult imported = null;
            try {
                if (request == IMPORT_MOD) {
                    imported = ModFiles.importArchive(getContentResolver(), uri, data);
                    done = (MenuLanguage.usesChinese(this) ? "已添加 MOD：" : "Mod added: ") + imported.description;
                }
                else if (request == IMPORT_CHEATS) done = "Cheats added: " + ModFiles.importCheats(getContentResolver(), uri, data);
                else if (request == IMPORT_DRIVER) done = "Driver added: " + Drivers.unpack(this, uri) + ". Choose it in the list to use it.";
                else done = "Lossless.dll imported (" + Drivers.importDll(this, uri, data) + "). It's used from the next start.";
            } catch (java.io.IOException | RuntimeException e) {
                done = "Couldn't add it: " + e.getMessage();
            }
            final ModFiles.ImportResult added = imported;
            final String msg = done;
            runOnUiThread(() -> {
                boolean restart = added != null && modsChanged(added.gameFiles, added.scripts);
                String reload = MenuLanguage.usesChinese(this) ? "。已请求重新加载脚本。" : ". Script reload requested.";
                Toast.makeText(this, msg + (added != null && added.scripts && !restart ? reload : ""),
                        Toast.LENGTH_LONG).show();
                if (isFinishing()) return;
                SettingsMenu menu = new SettingsMenu(this, controls);
                menu.show();
                if (restart) menu.askRestart();
            });
        }, "ModImport").start();
    }

    // ---- Settings → Saves (SaveFiles): pick a zip or a folder, maybe ask for the Steam ID, then where a zip goes --------

    private SaveFiles.Zip outgoing;  // a zip made, waiting for the picker to say where it goes

    // mode: 0 import, 1 convert, 2 only learn the Steam account
    void pickSave(boolean folder, int mode) {
        if (folder) start(new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE),
                mode == 1 ? CONVERT_FOLDER : mode == 2 ? LEARN_FOLDER : SAVE_FOLDER);
        else start(new Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("*/*")
                .putExtra(Intent.EXTRA_MIME_TYPES, new String[] {"application/zip", "application/x-zip-compressed",
                        "application/octet-stream"}), mode == 1 ? CONVERT_ZIP : mode == 2 ? LEARN_ZIP : SAVE_ZIP);
    }

    void exportSave(boolean steam) {
        File data = SettingsMenu.gameFolder(this);
        if (data == null) return;
        Toast.makeText(this, steam ? "Converting…" : "Packing…", Toast.LENGTH_SHORT).show();
        new Thread(() -> {
            try {
                SaveFiles.Zip z = steam ? SaveFiles.exportSteam(this, data) : SaveFiles.exportSwitch(data);
                runOnUiThread(() -> saveZip(z));
            } catch (java.io.IOException | RuntimeException e) {
                saveDone("Couldn't export it: " + e.getMessage());
            }
        }, "SaveExport").start();
    }

    private void saveZip(SaveFiles.Zip z) {
        outgoing = z;
        start(new Intent(Intent.ACTION_CREATE_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("application/zip")
                .putExtra(Intent.EXTRA_TITLE, z.name), SAVE_OUT);
    }

    private void savePicked(int request, Uri uri) {
        File data = SettingsMenu.gameFolder(this);
        if (uri == null || data == null) {
            new SettingsMenu(this, controls).show();
            return;
        }
        Toast.makeText(this, "Reading the save…", Toast.LENGTH_SHORT).show();
        new Thread(() -> {
            try {
                SaveFiles.Incoming in = request == SAVE_FOLDER || request == CONVERT_FOLDER || request == LEARN_FOLDER
                        ? SaveFiles.readFolder(getContentResolver(), uri) : SaveFiles.readZip(getContentResolver(), uri);
                saveWork(in, request == CONVERT_ZIP || request == CONVERT_FOLDER ? 1
                        : request == LEARN_ZIP || request == LEARN_FOLDER ? 2 : 0, 0);
            } catch (java.io.IOException | RuntimeException e) {
                saveDone("Couldn't read it: " + e.getMessage());
            }
        }, "SaveImport").start();
    }

    // On a background thread. A Steam save whose ID isn't known asks for it, and comes back here with it.
    private void saveWork(SaveFiles.Incoming in, int mode, long typed) {
        File data = SettingsMenu.gameFolder(this);
        try {
            if (mode == 1) {
                SaveFiles.Zip z = SaveFiles.convert(this, in, data, typed);
                runOnUiThread(() -> saveZip(z));
            } else if (mode == 2) {
                saveDone(SaveFiles.learnSteam(this, in, data, typed));
            } else {
                String done = SaveFiles.install(this, in, data, typed);
                runOnUiThread(() -> saveOver(done));
            }
        } catch (SaveData.Citrus.NeedId e) {
            runOnUiThread(() -> askSteamId(in, mode, typed != 0
                    ? "That Steam ID doesn't open this save. It's the account the save was made on." : ""));
        } catch (java.io.IOException | RuntimeException e) {
            saveDone((mode == 1 ? "Couldn't convert it: " : mode == 2 ? "Couldn't read it: " : "Couldn't import it: ")
                    + e.getMessage());
        }
    }

    // The ID wasn't in the save (only its SS1_* files and an emulator's settings have it): asked for
    private void askSteamId(SaveFiles.Incoming in, int mode, String error) {
        new TextPrompt(this).show("Steam ID of the save's account", "17 digits, starting with 7656", "", error, 0, 24, false,
                false, text -> {
                    if (text == null) {
                        saveDone(null);
                        return;
                    }
                    long id = SaveData.steamId(text);
                    if (id == 0) {
                        askSteamId(in, mode, "That isn't a Steam ID. It's the 17-digit number on your Steam profile's "
                                + "page (steamid.io finds it from the profile's link).");
                        return;
                    }
                    Toast.makeText(this, "Decrypting…", Toast.LENGTH_SHORT).show();
                    new Thread(() -> saveWork(in, mode, id), "SaveImport").start();
                });
    }

    // The menu again, on Saves, with what happened
    private void saveDone(String message) {
        runOnUiThread(() -> {
            if (message != null) SettingsMenu.saveMessage = message;
            if (!isFinishing()) new SettingsMenu(this, controls).show();
        });
    }

    // A save was put in place of the one the game has loaded. The game would write its own back the next time it saves
    // (an autosave is enough), so it closes now, without saving; the message is shown on the next start.
    void saveOver(String message) {
        File data = SettingsMenu.gameFolder(this);
        try {
            if (data != null) java.nio.file.Files.write(SaveFiles.pendingMessage(data).toPath(),
                    message.getBytes(java.nio.charset.StandardCharsets.UTF_8));
        } catch (java.io.IOException ignored) {
            // (only the message is lost)
        }
        Toast.makeText(this, "Save in place. The game closes now; open it again to play it.", Toast.LENGTH_LONG).show();
        getWindow().getDecorView().postDelayed(() -> {
            finishAffinity();
            android.os.Process.killProcess(android.os.Process.myPid());
        }, 2500);
    }

    // What the last import or restore said, once
    private void showSaveMessage() {
        File data = SettingsMenu.gameFolder(this);
        if (data == null) return;
        File f = SaveFiles.pendingMessage(data);
        try {
            String m = new String(java.nio.file.Files.readAllBytes(f.toPath()), java.nio.charset.StandardCharsets.UTF_8);
            f.delete();
            Toast.makeText(this, m, Toast.LENGTH_LONG).show();
        } catch (java.io.IOException ignored) {
            // (nothing imported)
        }
    }

    private void start(Intent i, int request) {
        pad.releaseAll();
        try {
            startActivityForResult(i, request);
        } catch (android.content.ActivityNotFoundException e) {
            Toast.makeText(this, "No file picker on this phone", Toast.LENGTH_LONG).show();
        }
    }

    @Override
    protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        Uri uri = result == RESULT_OK && data != null ? data.getData() : null;
        if (request == IMPORT_MOD || request == IMPORT_CHEATS || request == IMPORT_DRIVER || request == IMPORT_DLL) {
            importInto(request, uri);
        } else if (request == SAVE_ZIP || request == SAVE_FOLDER || request == CONVERT_ZIP || request == CONVERT_FOLDER
                || request == LEARN_ZIP || request == LEARN_FOLDER) {
            savePicked(request, uri);
        } else if (request == SAVE_OUT) {
            SaveFiles.Zip z = outgoing;
            outgoing = null;
            if (uri == null || z == null) {
                saveDone(null);
                return;
            }
            new Thread(() -> {
                try {
                    SaveFiles.write(getContentResolver(), uri, z.bytes);
                    saveDone(z.message);
                } catch (java.io.IOException | RuntimeException e) {
                    saveDone("Couldn't write the zip: " + e.getMessage());
                }
            }, "SaveExport").start();
        } else if (request == PICK_IMAGE) {
            if (uri != null && imageFor != null) {
                try {
                    controls.setImage(imageFor, LayoutFiles.readImage(getContentResolver(), uri));
                } catch (java.io.IOException | RuntimeException e) {
                    Toast.makeText(this, "Couldn't use that image: " + e.getMessage(), Toast.LENGTH_LONG).show();
                }
                refresh();
            }
            Runnable after = afterPick;
            imageFor = null;
            afterPick = null;
            if (after != null) after.run();
        } else if ((request == EXPORT_LAYOUT || request == IMPORT_LAYOUT) && uri != null) {
            try {
                if (request == EXPORT_LAYOUT) {
                    LayoutFiles.export(controls, getContentResolver(), uri);
                    Toast.makeText(this, "Layout exported", Toast.LENGTH_SHORT).show();
                } else {
                    LayoutFiles.importFrom(controls, getContentResolver(), uri);
                    controls.saveNow();
                    refresh();
                    Toast.makeText(this, "Layout imported", Toast.LENGTH_SHORT).show();
                }
            } catch (java.io.IOException | RuntimeException e) {
                Toast.makeText(this, (request == EXPORT_LAYOUT ? "Export failed: " : "Import failed: ") + e.getMessage(),
                        Toast.LENGTH_LONG).show();
            }
        }
    }

    @Override
    public void editControl(Controls.Ctl c) {
        new SettingsMenu(this, controls).editControl(c);
    }

    @Override
    public void addButton(int k) {
        new SettingsMenu(this, controls).editCombo(k);
    }
}
