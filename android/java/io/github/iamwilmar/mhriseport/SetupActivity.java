package io.github.iamwilmar.mhriseport;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.SharedPreferences;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.StateListDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.storage.StorageManager;
import android.os.storage.StorageVolume;
import android.provider.Settings;
import android.text.SpannableStringBuilder;
import android.text.Spanned;
import android.text.style.ForegroundColorSpan;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

// The start screen. The first time it asks for the files permission and two folders:
//   - the game folder, where the user copied the game files (ExeFS and RomFS already extracted on their own console, and
//     any DLC). DumpScanner checks what's there; the app only reads it, so it can be anywhere, even read-only.
//   - the data folder, the port's own: saves, the shader cache, log.txt, the settings switches. It can be the game folder
//     itself (how older versions worked) or any other. Where the game files are goes in its rutas.txt (rutas.h).
// files/game_folder.txt and files/carpeta.txt remember them; the game (GameActivity, a NativeActivity: libnier.so) reads
// the data folder from carpeta.txt. Without game_folder.txt (an install from before), the game folder is the data folder.
// After that, straight to the game. The app decrypts nothing and takes no NSP/XCI.
// Looks like the in-game settings menu: dark wine, bronze plates, gold headings. Every page is the same frame: a title
// and a line under it, the page itself (scrolls), a status line and the buttons at the bottom, above the system's bar.
// With a controller: the d-pad moves the focus and A presses (Android turns A into DPAD_CENTER).
public final class SetupActivity extends Activity {
    static final class MenuLanguage {
        static final String[] NAMES = {"English", "简体中文"};

        static boolean usesChinese(android.content.Context context) {
            return "zh-Hans".equals(context.getSharedPreferences("menu", 0).getString("language", "en"));
        }

        static void setChinese(android.content.Context context, boolean chinese) {
            context.getSharedPreferences("menu", 0).edit().putString("language", chinese ? "zh-Hans" : "en").apply();
        }
    }


    private static final String GAME_FILE = "game_folder.txt", DATA_FILE = "carpeta.txt";

    private static final int WINE = 0xFF26120E, PANEL = 0xFF1A0C09, ROW = 0xFF2C1614, LIT = 0xFF4A2A18;
    private static final int BRONZE = 0xFF3E2616, EDGE = SettingsMenu.EDGE, GOLD = SettingsMenu.GOLD;
    private static final int CREAM = SettingsMenu.CREAM, DIM = 0xFFB8A888, GOOD = 0xFF9CD38A, ALERT = 0xFFFF9678;

    private File browsing;      // the browser's folder; null: the list of storages
    private boolean forData;    // what the browser is choosing
    private boolean inBrowser;
    private boolean shown, waitingAccess;
    private boolean chinese;
    private DumpScanner.Result found;  // the game files of the game folder, once scanned

    private LinearLayout body, buttons;
    private TextView status;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        chinese = MenuLanguage.usesChinese(this);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (!Environment.isExternalStorageManager()) {
            showPermission();
            shown = false;
            return;
        }
        if (shown) return;  // coming back from the background doesn't change the screen
        shown = true;
        File data = saved(DATA_FILE);
        DumpScanner.Result saved = data != null ? DumpScanner.load(data) : null;
        if (saved != null) {
            found = saved;
            // Both folders chosen and the files where they were: straight to the game, past this screen. It shows again
            // if something is missing (below), if the game's settings panel asked for it ("Change folder" leaves the file
            // elegir_carpeta in the data folder) or if the app is opened twice within 10 seconds, the way out when the
            // game doesn't get as far as starting.
            File ask = new File(data, "elegir_carpeta");
            File stamp = new File(getFilesDir(), "ultimo_arranque.txt");
            long last = 0;
            try {
                last = Long.parseLong(new String(Files.readAllBytes(stamp.toPath())).trim());
            } catch (IOException | NumberFormatException e) {
                // no earlier start
            }
            long now = System.currentTimeMillis();
            boolean again = now >= last && now - last < 10000;
            if (saved.ready() && !ask.exists() && !again) {
                try {
                    Files.write(stamp.toPath(), Long.toString(now).getBytes());
                } catch (IOException e) {
                    // without the stamp only the "twice in a row" way out is lost
                }
                startGame();
                return;
            }
            ask.delete();
            stamp.delete();
            showResult();
            return;
        }
        File game = gameFolder();
        if (game != null && game.isDirectory()) scan(game);  // something moved since last time
        else if (waitingAccess) showBrowser(null, false);  // (back from the permission screen with it on)
        else showWelcome();
        waitingAccess = false;
    }

    @Override
    @SuppressWarnings("deprecation")  // still delivered with targetSdk 33; a controller sends BACK with B
    public void onBackPressed() {
        if (inBrowser && browsing != null) showBrowser(parentOf(browsing), forData);
        else if (inBrowser && found != null && found.ready()) showResult();
        else super.onBackPressed();
    }

    // ---- the pages ------------------------------------------------------------------------------------------------------

    private String tr(String english, String simplifiedChinese) {
        return chinese ? simplifiedChinese : english;
    }

    private void showPermission() {
        inBrowser = false;
        page(tr("FILE ACCESS", "文件访问权限"),
                tr("Allow file access in Android settings, then return here to choose your game folder.",
                   "请在系统设置中开启“允许访问所有文件”，再返回此处选择游戏文件夹。"), this::showPermission);
        gameGuide();
        Button grant = action(tr("Open permission settings", "打开权限设置"), true, v -> {
            waitingAccess = true;
            say(tr("Allow access to all files, then return here.", "开启“允许访问所有文件”后返回此处。"), GOLD);
            try {
                startActivity(new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                        Uri.parse("package:" + getPackageName())));
            } catch (ActivityNotFoundException e) {
                startActivity(new Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION));
            }
        });
        action(tr("Exit", "退出"), false, v -> finish());
        grant.requestFocus();
    }

    private void showWelcome() {
        inBrowser = false;
        page(tr("GAME FILES", "游戏文件"),
                tr("Choose the folder containing your extracted ExeFS and RomFS. MHRise reads the files in place.",
                   "请选择包含已提取的 ExeFS 和 RomFS 的文件夹。MHRise 会直接从这里读取游戏。"), this::showWelcome);
        gameGuide();
        Button pick = action(tr("Choose game folder", "选择游戏文件夹"), true, v -> showBrowser(null, false));
        action(tr("GPU driver", "GPU 驱动"), false, v -> startActivity(new Intent(this, DriverActivity.class)));
        action(tr("Exit", "退出"), false, v -> finish());
        pick.requestFocus();
    }

    // What the game folder has to look like, and what each part is
    private void gameGuide() {
        LinearLayout cols = columns();
        LinearLayout left = panel(cols, tr("EXAMPLE FOLDER", "游戏文件夹示例"));
        left.addView(tree(new String[] {
            tr("Your game folder", "游戏文件夹"),
            "├─ exefs",
            "│    main, main.npdm",
            "│    rtld, sdk, subsdk0",
            "├─ romfs",
            "│    " + Game.ROMFS_MARKER,
            "│    … " + Game.UPDATE_MARKER,
            tr("└─ DLC (optional)", "└─ DLC（可选）"),
            tr("     <title ID>/romfs", "     <标题 ID>/romfs"),
        }, 0, 1, 4, 7));
        LinearLayout right = column(cols, tr("GAME VERSION " + Game.VERSION, "游戏版本 " + Game.VERSION));
        right.addView(notes(tr("Use the ExeFS from this update and the base RomFS with the update applied.\n\n"
                        + "Setup searches the folder you choose and its subfolders. Keep each DLC’s title ID in its folder name.\n\n"
                        + "Next, choose a writable folder for saves and settings.",
                "请使用此版本的 ExeFS，以及已合并此更新的完整 RomFS。\n\n"
                        + "程序会检查所选文件夹及其子文件夹。若有 DLC，请保留文件夹名称中的标题 ID。\n\n"
                        + "下一步将选择可写入的数据文件夹，用于存放存档和设置。")));
    }

    // ---- the folder browser ---------------------------------------------------------------------------------------------

    private void showBrowser(File dir, boolean data) {
        inBrowser = true;
        browsing = dir;
        forData = data;
        page(data ? tr("DATA FOLDER", "数据文件夹") : tr("CHOOSE YOUR GAME FOLDER", "选择游戏文件夹"),
                data ? tr("Choose a writable folder for your saves, shader cache, log and settings.",
                          "请选择一个可写入的文件夹，用于存放存档、着色器缓存、日志和设置。")
                     : tr("Open the folder containing ExeFS and RomFS, then select “Use this folder”.",
                          "打开包含 ExeFS 和 RomFS 的文件夹，然后选择“使用此文件夹”。"),
                () -> showBrowser(dir, data));
        LinearLayout where = panel(body, null);
        TextView path = text(dir == null ? tr("Storage", "存储设备") : dir.getPath(), 14, GOLD, true);
        path.setTypeface(Typeface.MONOSPACE, Typeface.BOLD);
        where.addView(path);
        if (dir == null) {
            for (StorageVolume volume : getSystemService(StorageManager.class).getStorageVolumes()) {
                File d = volume.getDirectory();
                if (d != null) item(where, "💾   " + volume.getDescription(this), v -> showBrowser(d, data));
            }
        } else {
            item(where, tr("⬆   Up", "⬆   上一级"), v -> showBrowser(parentOf(dir), data));
            File[] subs = dir.listFiles(File::isDirectory);
            int n = 0;
            if (subs != null) {
                Arrays.sort(subs, (a, b) -> a.getName().compareToIgnoreCase(b.getName()));
                for (File sub : subs) {
                    if (sub.getName().startsWith(".")) continue;
                    item(where, "📁   " + sub.getName(), v -> showBrowser(sub, data));
                    n++;
                }
            }
            if (n == 0) where.addView(notes(tr("No subfolders", "没有子文件夹")));
        }
        Button first = null;
        if (dir != null) {
            first = action(tr("Use this folder", "使用此文件夹"), true, v -> {
                if (data) useDataFolder(dir);
                else scan(dir);
            });
            if (data) action(tr("New folder here", "新建文件夹"), false, v -> {
                File fresh = new File(dir, "MH Rise port");
                for (int k = 2; fresh.exists() && !fresh.isDirectory(); k++) fresh = new File(dir, "MH Rise port " + k);
                if (fresh.isDirectory() || fresh.mkdirs()) showBrowser(fresh, true);
                else say(tr("Could not create a folder in ", "无法在此位置新建文件夹：") + dir.getPath(), ALERT);
            });
        }
        boolean back = found != null && found.ready();
        action(back ? tr("Cancel", "取消") : tr("Exit", "退出"), false, v -> {
            if (back) showResult();
            else finish();
        });
        if (where.getChildCount() > 1) where.getChildAt(1).requestFocus();
        else if (first != null) first.requestFocus();
    }

    private void showBusy(String title, String what) {
        inBrowser = false;
        page(title, what, null);
        LinearLayout p = panel(body, null);
        p.addView(new android.widget.ProgressBar(this));
    }

    // ---- the game folder ------------------------------------------------------------------------------------------------

    private void scan(File dir) {
        showBusy(tr("CHECKING GAME FILES…", "正在检查游戏文件…"), dir.getPath());
        new Thread(() -> {
            DumpScanner.Result r = DumpScanner.scan(dir);
            String error = null;
            File data = saved(DATA_FILE);
            if (r.ready()) {
                try {
                    write(GAME_FILE, dir);
                    // (an install from before kept everything in the game folder: that stays its data folder)
                    if (data != null && data.isDirectory()) DumpScanner.writePaths(data, r);
                } catch (IOException e) {
                    error = e.toString();
                }
            }
            String failure = error;
            runOnUiThread(() -> {
                found = r;
                if (r.ready() && failure == null && (data == null || !data.isDirectory())) {
                    askDataFolder(dir);
                    return;
                }
                if (failure != null) showResult(() -> tr("Could not save the folder selection. ", "无法保存所选文件夹。") + failure, ALERT);
                else showResult();
            });
        }, "FindGame").start();
    }

    // The first time: the game files are fine, now where the port's own files go
    private void askDataFolder(File game) {
        inBrowser = false;
        page(tr("SAVES AND SETTINGS", "存档和设置"),
                tr("Game files verified. Choose where to keep your saves and settings.",
                   "游戏文件检查通过。请选择存放存档和设置的位置。"), () -> askDataFolder(game));
        LinearLayout cols = columns();
        LinearLayout left = panel(cols, tr("DATA FOLDER CONTENTS", "数据文件夹内容"));
        left.addView(tree(new String[] {
            tr("Your data folder", "数据文件夹"),
            tr("├─ save          saved game", "├─ save          游戏存档"),
            tr("├─ save.bak      save backup", "├─ save.bak      存档备份"),
            tr("├─ shaders       shader cache", "├─ shaders       着色器缓存"),
            "├─ log.txt",
            tr("└─ mhr.ini       settings", "└─ mhr.ini       设置"),
        }, 0));
        LinearLayout right = column(cols, tr("WHERE TO KEEP SAVES", "存放位置"));
        right.addView(notes(tr("A separate data folder makes saves easier to back up and lets you keep game files on a read-only card. "
                        + "You can also use the game folder if it is writable.",
                "单独存放便于备份存档，游戏文件也可以放在只读存储卡上。如果游戏文件夹可写，也可以将两者放在一起。")));
        say("✓  " + game.getPath(), GOOD);
        Button other = action(tr("Choose data folder", "选择数据文件夹"), true, v -> showBrowser(game.getParentFile(), true));
        action(tr("Use the game folder", "使用游戏文件夹"), false, v -> useDataFolder(game));
        other.requestFocus();
    }

    // ---- the data folder ------------------------------------------------------------------------------------------------

    private void useDataFolder(File dir) {
        File old = saved(DATA_FILE);
        if (old != null && (sameFolder(old, dir) || !old.isDirectory())) old = null;
        showBusy(old != null ? tr("COPYING SAVES…", "正在复制存档…") : tr("SETTING UP…", "正在设置…"), dir.getPath());
        final File from = old;
        final DumpScanner.Result r = found;
        new Thread(() -> {
            String error = null;
            Supplier<String> note = null;
            try {
                if (r == null || !r.ready()) throw new IOException(tr("Choose a valid game folder first.", "请先选择有效的游戏文件夹。"));
                if (!dir.isDirectory() && !dir.mkdirs()) throw new IOException(tr("Could not create the folder.", "无法创建文件夹。"));
                if (from != null) note = copyPortFiles(from, dir, r);
                DumpScanner.writePaths(dir, r);
                write(DATA_FILE, dir);
            } catch (IOException e) {
                error = e.toString();
            }
            String failure = error;
            Supplier<String> done = note;
            runOnUiThread(() -> {
                if (failure != null) showResult(() -> tr("Could not use ", "无法使用文件夹 ") + dir + " — " + failure, ALERT);
                else showResult(done, GOLD);
            });
        }, "DataFolder").start();
    }

    // The old data folder's contents, copied (not moved) to the new one: everything except the game files themselves.
    // A folder that already has saves keeps them: nothing is copied over them.
    private Supplier<String> copyPortFiles(File from, File to, DumpScanner.Result r) throws IOException {
        if (new File(to, "save").isDirectory())
            return () -> tr("Using the saves already in ", "已使用这里的现有存档：") + to;
        List<String> game = new ArrayList<>();
        game.add(r.exefs);
        game.add(r.romfs);
        game.addAll(r.dlc.values());
        File[] items = from.listFiles();
        if (items == null) return null;
        int copied = 0;
        for (File item : items) {
            String name = item.getName();
            if (name.equals("rutas.txt") || name.equals("rutas.txt.tmp") || name.equals("elegir_carpeta")) continue;
            if (holdsGame(item, game)) continue;
            copy(item, new File(to, name));
            copied++;
        }
        return copied == 0 ? null
                : () -> tr("Saves and settings copied. The originals are in ", "存档和设置已复制。原文件仍位于 ") + from;
    }

    // Is this a game folder, or one with game files somewhere inside?
    private static boolean holdsGame(File item, List<String> game) {
        String path = canonical(item);
        for (String g : game) {
            if (g == null) continue;
            String gp = canonical(new File(g));
            if (gp.equals(path) || gp.startsWith(path + File.separator)) return true;
        }
        return false;
    }

    private static void copy(File from, File to) throws IOException {
        if (from.isDirectory()) {
            if (!to.isDirectory() && !to.mkdirs()) throw new IOException("can't make " + to);
            File[] items = from.listFiles();
            if (items == null) return;
            for (File item : items) copy(item, new File(to, item.getName()));
        } else {
            Files.copy(from.toPath(), to.toPath(), StandardCopyOption.REPLACE_EXISTING,
                    StandardCopyOption.COPY_ATTRIBUTES);
        }
    }

    // ---- what's chosen --------------------------------------------------------------------------------------------------

    private void showResult() {
        showResult(null, GOLD);
    }

    private void showResult(Supplier<String> message, int color) {
        inBrowser = false;
        browsing = null;
        DumpScanner.Result r = found != null ? found : new DumpScanner.Result();
        File game = gameFolder(), data = saved(DATA_FILE);
        boolean dataOk = data != null && data.isDirectory();
        boolean ready = r.ready() && dataOk;
        page(Game.NAME.toUpperCase(java.util.Locale.ROOT),
                ready ? tr("Ready to start.", "可以启动游戏。") : tr("Check the items marked below.", "请检查下方标记的项目。"),
                () -> showResult(message, color));
        LinearLayout cols = columns();

        LinearLayout left = panel(cols, tr("GAME FOLDER", "游戏文件夹"));
        left.addView(pathLine(game != null ? game.getPath() : tr("None selected", "尚未选择")));
        left.addView(check(r.exefs != null, r.exefs != null ? tr("Game version ", "游戏版本 ") + Game.VERSION
                : r.exefsProblem != null ? r.exefsProblem.text(chinese)
                : tr("ExeFS not found. Look for the folder with main and main.npdm.", "未找到 ExeFS。请检查是否包含 main 和 main.npdm。")));
        left.addView(check(r.romfs != null, r.romfs != null ? tr("Game RomFS", "游戏 RomFS")
                : r.romfsProblem != null ? r.romfsProblem.text(chinese) : tr("RomFS not found.", "未找到 RomFS。")));
        if (Game.DLC_MARKER != null) {
            left.addView(r.dlc.isEmpty() ? line(tr("·  No DLC (optional)", "·  未找到 DLC（可选）"), DIM)
                    : check(true, r.dlc.size() + tr(" DLC", " 个 DLC")));
            if (r.unknownDlc > 0)
                left.addView(line("·  " + r.unknownDlc + tr(" DLC skipped. Keep the title ID in each folder name.",
                        " 个 DLC 已跳过。请保留文件夹名称中的标题 ID。"), DIM));
        }

        LinearLayout right = panel(cols, tr("DATA FOLDER", "数据文件夹"));
        right.addView(pathLine(dataOk ? data.getPath() : tr("None selected", "尚未选择")));
        if (dataOk) {
            boolean same = game != null && sameFolder(game, data);
            right.addView(check(true, same ? tr("Saves, cache and log alongside game files", "存档、缓存和日志与游戏文件一起存放")
                    : tr("Saves, cache and log", "存档、缓存和日志")));
            right.addView(line(new File(data, "save").isDirectory() ? tr("·  Has saves", "·  已有存档")
                    : tr("·  No saves yet", "·  暂无存档"), DIM));
        } else {
            right.addView(check(false, tr("Choose a folder for saves.", "请选择存档的存放位置。")));
        }

        Button first = null;
        if (ready) first = action(tr("Start game", "启动游戏"), true, v -> startGame());
        Button changeGame = action(r.ready() ? tr("Game folder", "游戏文件夹") : tr("Choose game folder", "选择游戏文件夹"), !r.ready(),
                v -> showBrowser(game != null && game.isDirectory() ? game : null, false));
        if (first == null && !r.ready()) first = changeGame;
        if (r.ready()) {
            Button changeData = action(dataOk ? tr("Data folder", "数据文件夹") : tr("Choose data folder", "选择数据文件夹"), !dataOk,
                    v -> showBrowser(dataOk ? data : null, true));
            if (first == null) first = changeData;
        }
        action(tr("GPU driver", "GPU 驱动"), false, v -> startActivity(new Intent(this, DriverActivity.class)));
        if (message != null) say(message.get(), color);
        first.requestFocus();
    }

    private void startGame() {
        File data = saved(DATA_FILE);
        if (data != null) {
            String migration = "scissoredClears:" + data.getAbsolutePath();
            SharedPreferences prefs = getPreferences(MODE_PRIVATE);
            if (!prefs.getBoolean(migration, false)) {
                if (!new Ini(data).removeSystemMemoryOverride()) {
                    showResult(() -> tr("Could not update TU_DEBUG in ", "无法更新 TU_DEBUG，文件为 ") + new File(data, Ini.NAME), ALERT);
                    return;
                }
                prefs.edit().putBoolean(migration, true).apply();
            }
            try {
                ModFiles.prepareForLaunch(data);
            } catch (IOException e) {
                showResult();
                android.widget.Toast.makeText(this, tr("Could not prepare PC mod patches: ", "无法准备 PC MOD 补丁：") + e.getMessage(),
                        android.widget.Toast.LENGTH_LONG).show();
                return;
            }
        }
        startActivity(new Intent(this, GameActivity.class));
        overridePendingTransition(0, 0);
        finish();
    }

    // ---- files/ ---------------------------------------------------------------------------------------------------------

    private File saved(String name) {
        try {
            String path = new String(Files.readAllBytes(new File(getFilesDir(), name).toPath())).trim();
            return path.isEmpty() ? null : new File(path);
        } catch (IOException e) {
            return null;
        }
    }

    private File gameFolder() {
        File game = saved(GAME_FILE);
        return game != null ? game : saved(DATA_FILE);
    }

    private void write(String name, File dir) throws IOException {
        try (FileWriter w = new FileWriter(new File(getFilesDir(), name))) {
            w.write(dir.getPath() + "\n");
        }
    }

    private static String canonical(File f) {
        try {
            return f.getCanonicalPath();
        } catch (IOException e) {
            return f.getAbsolutePath();
        }
    }

    private static boolean sameFolder(File a, File b) {
        return canonical(a).equals(canonical(b));
    }

    // Up one folder; from a storage's root, back to the list of storages.
    private File parentOf(File dir) {
        if (dir == null) return null;
        for (StorageVolume volume : getSystemService(StorageManager.class).getStorageVolumes())
            if (dir.equals(volume.getDirectory())) return null;
        return dir.getParentFile();
    }

    // ---- the frame ------------------------------------------------------------------------------------------------------

    // A new page: title, the line under it, an empty body, the status line and an empty row of buttons
    private void page(String title, String intro, Runnable redraw) {
        int pad = dp(14);
        LinearLayout whole = new LinearLayout(this);
        whole.setOrientation(LinearLayout.VERTICAL);
        whole.setBackgroundColor(WINE);

        LinearLayout top = new LinearLayout(this);
        top.setOrientation(LinearLayout.VERTICAL);
        top.setPadding(pad * 2, pad, pad * 2, dp(8));
        LinearLayout heading = new LinearLayout(this);
        heading.setGravity(Gravity.CENTER_VERTICAL);
        TextView t = text(title, 20, GOLD, true);
        t.setTypeface(Typeface.create(Typeface.SERIF, Typeface.BOLD));
        heading.addView(t, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1));
        if (redraw != null) {
            Button language = new Button(this);
            language.setAllCaps(false);
            language.setText("⚙  Language / 语言");
            language.setTextColor(CREAM);
            language.setPadding(dp(12), 0, dp(12), 0);
            language.setMinHeight(dp(48));
            language.setBackground(states(BRONZE, LIT, 1, 6));
            language.setOnClickListener(v -> new AlertDialog.Builder(this)
                    .setTitle("Language / 语言")
                    .setSingleChoiceItems(MenuLanguage.NAMES, chinese ? 1 : 0, (dialog, choice) -> {
                        dialog.dismiss();
                        if (chinese == (choice == 1)) return;
                        chinese = choice == 1;
                        MenuLanguage.setChinese(this, chinese);
                        redraw.run();
                    }).show());
            heading.addView(language);
        }
        top.addView(heading);
        TextView i = text(intro, 14, CREAM, false);
        i.setGravity(Gravity.CENTER);
        i.setPadding(0, dp(4), 0, 0);
        top.addView(i);
        View rule = new View(this);
        rule.setBackgroundColor(EDGE);
        LinearLayout.LayoutParams rlp = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 1);
        rlp.topMargin = dp(10);
        top.addView(rule, rlp);
        whole.addView(top);

        ScrollView scroll = new ScrollView(this);
        body = new LinearLayout(this);
        body.setOrientation(LinearLayout.VERTICAL);
        body.setPadding(pad * 2, 0, pad * 2, pad);
        scroll.addView(body);
        whole.addView(scroll, new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1));

        status = text("", 14, GOLD, false);
        status.setGravity(Gravity.CENTER);
        status.setPadding(pad * 2, dp(6), pad * 2, dp(6));
        status.setVisibility(View.GONE);
        whole.addView(status);

        buttons = new LinearLayout(this);
        buttons.setGravity(Gravity.CENTER);
        buttons.setPadding(pad, 0, pad, dp(10));
        whole.addView(buttons);

        whole.setOnApplyWindowInsetsListener((v, insets) -> {  // (the buttons stay above the system's bar)
            v.setPadding(insets.getSystemWindowInsetLeft(), insets.getSystemWindowInsetTop(),
                    insets.getSystemWindowInsetRight(), insets.getSystemWindowInsetBottom());
            return insets;
        });
        setContentView(whole);
    }

    private void say(String s, int color) {
        status.setText(s);
        status.setTextColor(color);
        status.setVisibility(View.VISIBLE);
    }

    // A button at the bottom; the main one in gold
    private Button action(String label, boolean main, View.OnClickListener click) {
        Button b = new Button(this);
        b.setText(label);
        b.setAllCaps(false);
        b.setTextColor(main ? GOLD : CREAM);
        b.setTextSize(TypedValue.COMPLEX_UNIT_SP, 15);
        if (main) b.setTypeface(Typeface.DEFAULT_BOLD);
        b.setBackground(states(BRONZE, LIT, main ? 2 : 1, 6));
        b.setPadding(dp(18), 0, dp(18), 0);
        b.setOnClickListener(click);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, dp(46));
        lp.setMargins(dp(6), 0, dp(6), 0);
        buttons.addView(b, lp);
        return b;
    }

    // Two columns side by side in the body (a phone is in landscape here)
    private LinearLayout columns() {
        LinearLayout cols = new LinearLayout(this);
        cols.setOrientation(LinearLayout.HORIZONTAL);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        lp.topMargin = dp(10);
        body.addView(cols, lp);
        return cols;
    }

    // A dark plate with a gold heading (none: just the plate), in the body or in a column
    private LinearLayout panel(LinearLayout parent, String heading) {
        LinearLayout p = new LinearLayout(this);
        p.setOrientation(LinearLayout.VERTICAL);
        p.setBackground(plate(PANEL, 2, 6));
        p.setPadding(dp(14), dp(10), dp(14), dp(10));
        if (heading != null) p.addView(text(heading, 13, GOLD, true));
        LinearLayout.LayoutParams lp = parent == body
                ? new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
                : new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, 1);
        if (parent == body) lp.topMargin = dp(10);
        else if (parent.getChildCount() > 0) lp.leftMargin = dp(12);
        parent.addView(p, lp);
        return p;
    }

    // A column without a plate (the notes next to a panel)
    private LinearLayout column(LinearLayout parent, String heading) {
        LinearLayout c = new LinearLayout(this);
        c.setOrientation(LinearLayout.VERTICAL);
        c.setPadding(dp(18), dp(10), 0, 0);
        c.addView(text(heading, 13, GOLD, true));
        parent.addView(c, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1));
        return c;
    }

    // A folder in the browser: a row that lights up under the focus or the finger
    private void item(LinearLayout parent, String label, View.OnClickListener click) {
        TextView t = text(label, 15, CREAM, false);
        t.setPadding(dp(12), dp(10), dp(12), dp(10));
        t.setBackground(states(ROW, LIT, 1, 3));
        t.setFocusable(true);
        t.setClickable(true);
        t.setOnClickListener(click);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        lp.topMargin = dp(4);
        parent.addView(t, lp);
    }

    // A folder tree: folders (the lines given) in gold, what's in them in cream
    private TextView tree(String[] lines, int... folders) {
        SpannableStringBuilder b = new SpannableStringBuilder();
        for (int i = 0; i < lines.length; i++) {
            int at = b.length();
            b.append(lines[i]).append(i + 1 < lines.length ? "\n" : "");
            boolean folder = false;
            for (int f : folders) folder |= f == i;
            b.setSpan(new ForegroundColorSpan(folder ? GOLD : CREAM), at, b.length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        }
        TextView t = new TextView(this);
        t.setText(b);
        t.setTypeface(Typeface.MONOSPACE);
        t.setTextSize(TypedValue.COMPLEX_UNIT_SP, 13);
        t.setPadding(0, dp(6), 0, 0);
        return t;
    }

    private TextView notes(String s) {
        TextView t = text(s, 14, CREAM, false);
        t.setPadding(0, dp(6), 0, 0);
        t.setLineSpacing(dp(2), 1);
        return t;
    }

    private TextView pathLine(String s) {
        TextView t = text(s, 13, DIM, false);
        t.setTypeface(Typeface.MONOSPACE);
        t.setPadding(0, dp(4), 0, dp(4));
        return t;
    }

    private TextView check(boolean ok, String s) {
        return line((ok ? "✓  " : "✗  ") + s, ok ? GOOD : ALERT);
    }

    private TextView line(String s, int color) {
        TextView t = text(s, 14, color, false);
        t.setPadding(0, dp(4), 0, 0);
        return t;
    }

    private TextView text(String s, int sp, int color, boolean bold) {
        TextView t = new TextView(this);
        t.setText(s);
        t.setTextSize(TypedValue.COMPLEX_UNIT_SP, sp);
        t.setTextColor(color);
        if (bold) t.setTypeface(Typeface.DEFAULT_BOLD);
        return t;
    }

    private GradientDrawable plate(int fill, int strokeDp, int radiusDp) {
        GradientDrawable d = new GradientDrawable();
        d.setColor(fill);
        d.setStroke(dp(strokeDp), EDGE);
        d.setCornerRadius(dp(radiusDp));
        return d;
    }

    // Plain, and lit with a gold edge while focused (a controller) or pressed
    private StateListDrawable states(int fill, int lit, int strokeDp, int radiusDp) {
        StateListDrawable s = new StateListDrawable();
        GradientDrawable on = plate(lit, strokeDp + 1, radiusDp);
        on.setStroke(dp(strokeDp + 1), GOLD);
        s.addState(new int[] {android.R.attr.state_pressed}, on);
        s.addState(new int[] {android.R.attr.state_focused}, on);
        s.addState(new int[0], plate(fill, strokeDp, radiusDp));
        return s;
    }

    private int dp(int v) {
        return Math.round(TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, getResources().getDisplayMetrics()));
    }
}
