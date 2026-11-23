package io.github.iamwilmar.mhriseport;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.Process;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

import static io.github.iamwilmar.mhriseport.SettingsMenu.CREAM;
import static io.github.iamwilmar.mhriseport.SettingsMenu.DIM;
import static io.github.iamwilmar.mhriseport.SettingsMenu.EDGE;
import static io.github.iamwilmar.mhriseport.SettingsMenu.GOLD;
import static io.github.iamwilmar.mhriseport.SettingsMenu.GREEN;
import static io.github.iamwilmar.mhriseport.SettingsMenu.RED;
import static io.github.iamwilmar.mhriseport.SettingsMenu.ROW;

// "Driver de GPU": un driver Vulkan propio para una Adreno (Turnip, o una versión de Qualcomm más nueva), en vez del
// del sistema. Acepta los .zip de siempre de estos drivers (meta.json y la librería .so, como los de AdrenoTools). Cada uno
// se descomprime en files/drivers/<carpeta>/ (el linker no carga librerías de la SD) y el elegido se apunta en
// files/drivers/elegido, que lee el juego al arrancar (runtime/src/nnsdk/nvn/driver.h). Si con él no llega a dibujar, el
// juego vuelve solo al del sistema y deja files/drivers/fallo, que esta pantalla enseña y borra.
// Se abre desde la pantalla de inicio y desde el menú de ajustes del juego; el cambio vale desde el siguiente arranque.
public final class DriverActivity extends Activity {
    static final String FROM_GAME = "desde_juego";
    private static final int PICK = 1;
    private static final int PICK_DLL = 2;
    private static final int PANEL = 0xFF201010;

    private LinearLayout list;
    private TextView status;
    private String startedWith;  // el elegido al abrir: si cambia y venimos del juego, se ofrece cerrarlo

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(PANEL);

        LinearLayout top = new LinearLayout(this);
        top.setGravity(Gravity.CENTER_VERTICAL);
        top.setPadding(dp(18), dp(10), dp(10), dp(10));
        TextView title = text("SETTINGS", 22, GOLD, true);
        title.setTypeface(Typeface.create(Typeface.SERIF, Typeface.BOLD));
        TextView sub = text("  ◆  GPU driver", 17, CREAM, false);
        TextView close = text("✕", 22, CREAM, false);
        close.setPadding(dp(14), dp(4), dp(14), dp(4));
        close.setBackground(plate(0xFF1A0C09, 1, 2));
        close.setOnClickListener(v -> finish());
        top.addView(title);
        top.addView(sub, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        top.addView(close);
        root.addView(top);
        root.addView(rule(RED, 2));
        root.addView(rule(GREEN, 1));

        list = new LinearLayout(this);
        list.setOrientation(LinearLayout.VERTICAL);
        list.setPadding(dp(16), dp(6), dp(16), dp(16));
        ScrollView scroll = new ScrollView(this);
        scroll.addView(list);
        root.addView(scroll, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));
        setContentView(root);

        startedWith = selected()[0];
        build();
    }

    // ---- pantalla -------------------------------------------------------------------------------------------------------

    private void build() {
        list.removeAllViews();
        note("Adreno GPUs (Snapdragon) only, at your own risk. Add a driver .zip (meta.json and the .so library, like the "
                        + "Turnip ones for AdrenoTools) and pick it. It takes effect the next time the game starts; if the game "
                        + "can't draw with it, it goes back to the system's by itself.");
        status = text("", 14, CREAM, false);
        status.setPadding(dp(4), dp(6), dp(4), 0);
        status.setVisibility(View.GONE);
        list.addView(status);
        String failed = Drivers.takeFailure(this);
        if (failed != null) say("The last one didn't work: " + failed + ". The system's is in use.", true);

        String chosen = selected()[0];
        header("Drivers");
        entry("System driver", "The one that came with the phone",
                chosen == null, () -> choose(null), null);
        for (Drivers.Driver d : Drivers.list(this))
            entry(d.name, d.detail, d.dir.getName().equals(chosen), () -> choose(d.dir), () -> remove(d.dir));
        TextView add = button("Add a driver (.zip)", true);
        add.setOnClickListener(v -> pick());
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(50));
        lp.topMargin = dp(14);
        list.addView(add, lp);

        buildLossless();

        String now = selected()[0];
        boolean changed = now == null ? startedWith != null : !now.equals(startedWith);
        if (getIntent().getBooleanExtra(FROM_GAME, false) && changed) {
            note("The game keeps the driver it started with. Close it and open the app again to use the new one.");
            TextView quit = button("Close the game", false);
            quit.setOnClickListener(v -> {
                finishAffinity();
                Process.killProcess(Process.myPid());
            });
            LinearLayout.LayoutParams q = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(46));
            q.topMargin = dp(8);
            list.addView(quit, q);
        }
    }

    // Una fila: nombre y detalle, ✓ si es el elegido, y "Quitar" si se puede borrar
    private void entry(String name, String detail, boolean on, Runnable choose, Runnable remove) {
        LinearLayout r = new LinearLayout(this);
        r.setGravity(Gravity.CENTER_VERTICAL);
        r.setPadding(dp(14), dp(10), dp(8), dp(10));
        r.setBackground(plate(on ? 0xFF3E2616 : ROW, on ? 2 : 1, 2));
        TextView mark = text(on ? "✓" : "", 18, GOLD, true);
        mark.setGravity(Gravity.CENTER);
        r.addView(mark, new LinearLayout.LayoutParams(dp(26), ViewGroup.LayoutParams.WRAP_CONTENT));
        LinearLayout words = new LinearLayout(this);
        words.setOrientation(LinearLayout.VERTICAL);
        words.addView(text(name, 16, on ? GOLD : CREAM, on));
        if (detail != null && !detail.isEmpty()) words.addView(text(detail, 13, DIM, false));
        r.addView(words, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        if (remove != null) {
            TextView x = button("Remove", false);
            x.setOnClickListener(v -> remove.run());
            r.addView(x, new LinearLayout.LayoutParams(dp(92), dp(40)));
        }
        r.setOnClickListener(v -> choose.run());
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
        lp.topMargin = dp(4);
        list.addView(r, lp);
    }

    // A setting with a few choices: its name and what it is, the current choice on the right with a ▾, and a tap opens
    // the list (dropDown)
    private void choiceRow(String name, String value, String detail, Runnable open, Runnable remove) {
        LinearLayout r = new LinearLayout(this);
        r.setGravity(Gravity.CENTER_VERTICAL);
        r.setPadding(dp(14), dp(10), dp(8), dp(10));
        r.setBackground(plate(ROW, 1, 2));
        LinearLayout words = new LinearLayout(this);
        words.setOrientation(LinearLayout.VERTICAL);
        words.addView(text(name, 16, CREAM, false));
        if (detail != null && !detail.isEmpty()) words.addView(text(detail, 13, DIM, false));
        r.addView(words, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        TextView v = text(value + "  ▾", 16, GOLD, true);
        v.setPadding(dp(10), 0, dp(10), 0);
        r.addView(v);
        if (remove != null) {
            TextView x = button("Remove", false);
            x.setOnClickListener(c -> remove.run());
            r.addView(x, new LinearLayout.LayoutParams(dp(92), dp(40)));
        }
        r.setOnClickListener(c -> open.run());
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
        lp.topMargin = dp(4);
        list.addView(r, lp);
    }

    // The list of choices, as a panel like the settings menu's: the current one marked
    private void dropDown(String title, String[] options, int current, java.util.function.IntConsumer pick) {
        final android.app.Dialog d = new android.app.Dialog(this);
        d.requestWindowFeature(android.view.Window.FEATURE_NO_TITLE);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackground(plate(PANEL, 2, 0));
        root.setPadding(dp(10), dp(8), dp(10), dp(10));
        TextView t = text("◆  " + title, 18, GOLD, true);
        t.setTypeface(Typeface.create(Typeface.SERIF, Typeface.BOLD));
        t.setPadding(dp(6), dp(4), dp(6), dp(8));
        root.addView(t);
        for (int i = 0; i < options.length; i++) {
            final int k = i;
            final boolean on = i == current;
            TextView o = text((on ? "◆  " : "") + options[i], 16, on ? GOLD : CREAM, on);
            o.setPadding(dp(14), dp(12), dp(14), dp(12));
            o.setBackground(plate(on ? 0xFF3E2616 : ROW, on ? 2 : 1, 2));
            o.setOnClickListener(v -> {
                d.dismiss();
                if (k != current) pick.accept(k);
            });
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT);
            lp.topMargin = dp(4);
            root.addView(o, lp);
        }
        d.setContentView(root);
        android.view.Window w = d.getWindow();
        if (w != null) {
            w.setBackgroundDrawable(new android.graphics.drawable.ColorDrawable(android.graphics.Color.TRANSPARENT));
            w.setLayout(Math.min(dp(460), (int) (getResources().getDisplayMetrics().widthPixels * 0.6f)),
                    ViewGroup.LayoutParams.WRAP_CONTENT);
        }
        d.show();
    }

    private void say(String s, boolean bad) {
        status.setText(s);
        status.setTextColor(bad ? 0xFFE07060 : CREAM);
        status.setVisibility(View.VISIBLE);
    }

    // ---- elegir, quitar -------------------------------------------------------------------------------------------------

    private String[] selected() {  // (the folder name only; [1] is unused now)
        return new String[] {Drivers.chosen(this), null};
    }

    private void choose(File dir) {
        try {
            Drivers.choose(this, dir);
        } catch (IOException e) {
            say("Could not save it: " + e.getMessage(), true);
            return;
        }
        build();
    }

    private void remove(File dir) {
        try {
            Drivers.remove(this, dir);
        } catch (IOException e) {
            say("Could not remove it: " + e.getMessage(), true);
        }
        build();
    }

    // ---- Lossless Scaling -------------------------------------------------------------------------------------------------

    // Frame generation needs the player's own Lossless.dll (from Lossless Scaling on Steam; not shipped with the app).
    // It's copied to files/Lossless.dll, where the game looks first (runtime/src/nnsdk/nvn/lsfg/dll.cpp); the on/off
    // switch is the lsfg folder in the game folder, as before. The swapchain is sized at start-up, so both changes take
    // effect the next time the game starts.
    private File losslessDll() {
        return Drivers.losslessDll(this);
    }

    private File gameFolder() {
        try {
            String path = new String(Files.readAllBytes(new File(getFilesDir(), "carpeta.txt").toPath())).trim();
            return path.isEmpty() ? null : new File(path);
        } catch (IOException e) {
            return null;
        }
    }

    // Frame generation: Lossless Scaling's LSFG with the player's Lossless.dll, or the port's own generator (no DLL,
    // any Vulkan 1.1 GPU: runtime/src/nnsdk/nvn/lsfg/motion.h). The game uses LSFG when there's a DLL and it runs on
    // this GPU, the built-in one otherwise, or always with the fg_builtin switch (a folder in the game folder, as lsfg).
    private void buildLossless() {
        header("Frame generation");
        note("Shows about twice the game's fps (30 → 60) by making a picture between each two of the game's. With your own "
                + "Lossless.dll (from Lossless Scaling, Steam, Windows: in the program's folder; not included with the "
                + "app) it uses Lossless Scaling's frame generation; without it, or where that can't run, the port's own. "
                + "Takes effect the next time the game starts.");
        File dll = losslessDll();
        File folder = gameFolder();
        // (on and off in mhr.ini, like the in-game panel's row, which the game reads live)
        Ini ini = folder != null ? new Ini(folder) : null;
        File builtin = folder != null ? new File(folder, "fg_builtin") : null;
        boolean on = ini != null && ini.on("lsfg");
        boolean own = !dll.isFile() || (builtin != null && builtin.exists());
        entry("Frame generation", (on ? "On" : "Off") + " · " + (own ? "built-in" : "Lossless Scaling") + ". Tap to turn it "
                + (on ? "off." : "on."), on, () -> {
            if (ini == null) {
                say("Choose the game folder first.", true);
                return;
            }
            ini.set("lsfg", on ? "0" : "1");
            build();
        }, null);

        if (dll.isFile()) {
            String size = String.format(java.util.Locale.ROOT, "%.1f MB", dll.length() / 1048576.0);
            String[] kinds = {"Lossless Scaling", "Built-in"};
            choiceRow("Generator", kinds[own ? 1 : 0], "Lossless Scaling: sharper, heavier on the GPU (Lossless.dll "
                    + "imported, " + size + "). Built-in: lighter, runs on any GPU.", () -> dropDown("Generator", kinds,
                    own ? 1 : 0, k -> {
                        if (builtin == null) return;
                        if (k == 0) deleteTree(builtin);
                        else if (!builtin.isDirectory() && !builtin.mkdirs()) say("Couldn't write to the game folder.", true);
                        build();
                    }), () -> {
                dll.delete();
                build();
            });
        }
        if (folder != null) {
            // How finely the generator follows motion: Lossless Scaling's flow scale, or for the built-in one the size of
            // its finest flow level (a quarter, a third or half of the picture). Lower is lighter but softer and smears
            // more. The game picks it up at once.
            File half = new File(folder, "lsfg_flow_50"), full = new File(folder, "lsfg_flow_100");
            int level = full.exists() ? 2 : half.exists() ? 1 : 0;
            String[] names = {"Light", "Balanced", "Sharp"};
            String[] detail = own ? new String[] {"motion at 1/4", "motion at 1/3", "motion at 1/2"}
                                  : new String[] {"flow 25 %", "flow 50 %", "flow 100 %"};
            String[] options = new String[3];
            for (int i = 0; i < 3; i++) options[i] = names[i] + " (" + detail[i] + ")";
            choiceRow("Quality Preset", names[level], "How finely " + (own ? "the generator" : "Lossless Scaling")
                    + " follows motion. Higher is sharper with less smearing, and heavier on the GPU (each step about "
                    + "twice the work). Applies at once.", () -> dropDown("Quality Preset", options, level, k -> {
                        deleteTree(half);
                        deleteTree(full);
                        if (k == 1 && !half.mkdirs()) say("Couldn't write to the game folder.", true);
                        if (k == 2 && !full.mkdirs()) say("Couldn't write to the game folder.", true);
                        build();
                    }), null);
        }
        TextView add = button(dll.isFile() ? "Replace Lossless.dll" : "Import Lossless.dll", false);
        add.setOnClickListener(v -> pickDll());
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(50));
        lp.topMargin = dp(10);
        list.addView(add, lp);
    }

    private void pickDll() {
        Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        i.addCategory(Intent.CATEGORY_OPENABLE);
        i.setType("*/*");
        try {
            startActivityForResult(i, PICK_DLL);
        } catch (android.content.ActivityNotFoundException e) {
            say("There's no file picker on this phone.", true);
        }
    }

    private String importDll(Uri uri) throws IOException {
        return Drivers.importDll(this, uri, gameFolder());
    }

    // ---- agregar --------------------------------------------------------------------------------------------------------

    private void pick() {
        Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        i.addCategory(Intent.CATEGORY_OPENABLE);
        i.setType("*/*");
        i.putExtra(Intent.EXTRA_MIME_TYPES, new String[] {"application/zip", "application/x-zip-compressed",
                "application/octet-stream"});
        try {
            startActivityForResult(i, PICK);
        } catch (android.content.ActivityNotFoundException e) {
            say("There's no file picker on this phone.", true);
        }
    }

    @Override
    protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (result != RESULT_OK || data == null || data.getData() == null) return;
        if (request == PICK_DLL) {
            Uri dll = data.getData();
            say("Importing…", false);
            new Thread(() -> {
                String size = null, error = null;
                try {
                    size = importDll(dll);
                } catch (IOException e) {
                    error = e.getMessage();
                }
                final String ok = size, bad = error;
                runOnUiThread(() -> {
                    build();
                    if (ok != null) say("Lossless.dll imported (" + ok + "). Frame generation is on from the next start.", false);
                    else say("Couldn't import it: " + bad, true);
                });
            }, "ImportarLossless").start();
            return;
        }
        if (request != PICK) return;
        Uri zip = data.getData();
        say("Adding…", false);
        new Thread(() -> {
            String shown, error = null;
            try {
                shown = unpack(zip);
            } catch (IOException e) {
                shown = null;
                error = e.getMessage();
            }
            final String ok = shown, bad = error;
            runOnUiThread(() -> {
                build();
                if (ok != null) say("Added: " + ok + ". Tap it to use it.", false);
                else say("Couldn't add it: " + bad, true);
            });
        }, "AgregarDriver").start();
    }

    private String unpack(Uri zip) throws IOException {
        return Drivers.unpack(this, zip);
    }

    private static void deleteTree(File f) {
        Drivers.deleteTree(f);
    }

    // ---- piezas (las del menú de ajustes) -------------------------------------------------------------------------------

    private void header(String title) {
        TextView t = text(title, 18, GOLD, true);
        t.setTypeface(Typeface.create(Typeface.SERIF, Typeface.BOLD));
        t.setPadding(dp(4), dp(16), 0, dp(4));
        list.addView(t);
        list.addView(rule(EDGE, 1));
    }

    private void note(String s) {
        TextView t = text(s, 13, DIM, false);
        t.setPadding(dp(4), dp(8), dp(4), dp(2));
        list.addView(t);
    }

    private TextView button(String s, boolean main) {
        TextView b = text(s, 16, main ? 0xFF281010 : CREAM, true);
        b.setGravity(Gravity.CENTER);
        b.setBackground(plate(main ? GOLD : 0xFF3E2616, 1, 4));
        return b;
    }

    private View rule(int color, int heightDp) {
        View v = new View(this);
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
        TextView t = new TextView(this);
        t.setText(s);
        t.setTextSize(TypedValue.COMPLEX_UNIT_SP, sp * scale());
        t.setTextColor(color);
        if (bold) t.setTypeface(Typeface.DEFAULT_BOLD);
        return t;
    }

    private float scale = -1;  // (el tamaño del menú de ajustes, el mismo aquí)
    private float scale() {
        if (scale < 0) scale = SettingsMenu.sizeFactor(this);
        return scale;
    }

    private int dp(int v) {
        float px = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v * scale(), getResources().getDisplayMetrics());
        return v > 0 ? Math.max(1, Math.round(px)) : Math.round(px);
    }
}
