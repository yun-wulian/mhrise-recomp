package io.github.iamwilmar.mhriseport;

import android.content.Context;
import android.content.SharedPreferences;

// Los ajustes de los controles, guardados en las preferencias de la app ("controles"): los táctiles (si se ven, su
// opacidad y tamaño, la vibración, deslizar de un botón a otro, la zona muerta de sus sticks), dónde está cada control,
// los combos (un botón en pantalla, o uno del mando, que pulsa varios de la consola) y el mando físico (zona muerta,
// gatillos, dónde lee el stick derecho). TouchPad dibuja y lee los táctiles; lo del mando lo aplica el loader
// (GameActivity.nativePad).
final class Controls {
    // Bits de nn::hid::NpadButton, como en runtime/src/loader/android_main.cpp.
    static final long A = 1L, B = 1L << 1, X = 1L << 2, Y = 1L << 3, LSTICK = 1L << 4, RSTICK = 1L << 5, L = 1L << 6,
            R = 1L << 7, ZL = 1L << 8, ZR = 1L << 9, PLUS = 1L << 10, MINUS = 1L << 11, LEFT = 1L << 12, UP = 1L << 13,
            RIGHT = 1L << 14, DOWN = 1L << 15;
    // The sticks' directions (NpadButton has them too): in a combo or a mapped key one pushes its stick all the way
    // that way (the loader's force_sticks), over the finger's and the controller's
    static final long LS_LEFT = 1L << 16, LS_UP = 1L << 17, LS_RIGHT = 1L << 18, LS_DOWN = 1L << 19, RS_LEFT = 1L << 20,
            RS_UP = 1L << 21, RS_RIGHT = 1L << 22, RS_DOWN = 1L << 23;

    // Los botones que se pueden poner en un combo, en el orden del menú, y su nombre.
    static final long[] BUTTONS = {A, B, X, Y, L, R, ZL, ZR, PLUS, MINUS, LSTICK, RSTICK, UP, DOWN, LEFT, RIGHT,
            LS_UP, LS_DOWN, LS_LEFT, LS_RIGHT, RS_UP, RS_DOWN, RS_LEFT, RS_RIGHT};
    static final String[] NAMES = {"A", "B", "X", "Y", "L", "R", "ZL", "ZR", "+", "−", "L3", "R3", "↑", "↓", "←", "→",
            "LS↑", "LS↓", "LS←", "LS→", "RS↑", "RS↓", "RS←", "RS→"};

    static String names(long mask) {
        StringBuilder s = new StringBuilder();
        for (int i = 0; i < BUTTONS.length; i++) {
            if ((mask & BUTTONS[i]) == 0) continue;
            if (s.length() > 0) s.append('+');
            s.append(NAMES[i]);
        }
        return s.toString();
    }

    // Los controles en pantalla. Cada uno: dónde está su centro (fracción del ancho y del alto de la pantalla), su tamaño
    // (1: el suyo) y si se ve.
    enum Ctl {
        LSTICK, RSTICK, DPAD, A, B, X, Y, L, R, ZL, ZR, MINUS, PLUS, L3, R3, MENU, MOD, COMBO1, COMBO2, COMBO3, COMBO4,
        COMBO5, COMBO6, COMBO7, COMBO8;

        boolean combo() { return ordinal() >= COMBO1.ordinal(); }
        int comboIndex() { return ordinal() - COMBO1.ordinal(); }
    }

    // La disposición tradicional: la de un mando, con todo a la vista. Sticks arriba a la izquierda y abajo a la derecha
    // (como la consola), la cruceta debajo del izquierdo, los cuatro botones arriba a la derecha, los gatillos en las
    // esquinas de arriba, − y + arriba en el centro con el menú entre ellos, y L3 / R3 al lado de su stick.
    static final float[][] HOME = {
        {0.12f, 0.46f, 1}, // LSTICK
        {0.73f, 0.78f, 1}, // RSTICK
        {0.26f, 0.79f, 1}, // DPAD
        {0.935f, 0.46f, 1}, // A
        {0.875f, 0.60f, 1}, // B
        {0.875f, 0.32f, 1}, // X
        {0.815f, 0.46f, 1}, // Y
        {0.16f, 0.10f, 1}, // L
        {0.84f, 0.10f, 1}, // R
        {0.05f, 0.21f, 1}, // ZL
        {0.95f, 0.21f, 1}, // ZR
        {0.42f, 0.07f, 1}, // MINUS
        {0.58f, 0.07f, 1}, // PLUS
        {0.045f, 0.70f, 1}, // L3
        {0.585f, 0.90f, 1}, // R3
        {0.50f, 0.07f, 1}, // MENU
        {0.34f, 0.07f, 1}, // MOD (the MOD UI: only there with Lua scripts installed)
        {0.69f, 0.36f, 1}, // COMBO1
        {0.69f, 0.56f, 1}, // COMBO2
        {0.62f, 0.36f, 1}, // COMBO3
        {0.62f, 0.56f, 1}, // COMBO4
        {0.55f, 0.36f, 1}, // COMBO5
        {0.55f, 0.56f, 1}, // COMBO6
        {0.48f, 0.36f, 1}, // COMBO7
        {0.48f, 0.56f, 1}, // COMBO8
    };

    // Combos are also the layout editor's added buttons (Add, or Copy as a new button): every button on the screen
    // that isn't one of the console's is a combo, so a controller button can always fire it too.
    static final int COMBOS = 8;

    // A button's look in the layout editor: its shape (0 its own) and an icon instead of its letters (0 none)
    static final String[] SHAPES = {"Its own", "Circle", "Rounded square", "Square"};
    // Plate colours a button can have instead of the game's wine (0), and their fills
    static final String[] COLORS = {"Wine (the game's)", "Black", "Crimson", "Navy", "Forest", "Purple", "Slate", "Clear"};
    static final int[] COLOR_FILLS = {0, 0xFF101010, 0xFF8A1420, 0xFF14285A, 0xFF1E4A22, 0xFF4A1E5A, 0xFF3C4650, 0x01000000};  // (clear: no plate, the edge only)
    static final String[] ICONS = {"None (letters)", "Arrow up", "Arrow down", "Arrow left", "Arrow right", "Star",
            "Diamond", "Shield", "Blade", "Crosshair", "Dash"};
    // Los botones del mando que pueden disparar un combo, en el orden en que el menú los recorre (0: ninguno).
    static final long[] PAD_TRIGGERS = {0, LSTICK, RSTICK, LSTICK | RSTICK, MINUS, PLUS, MINUS | PLUS, L | R, ZL | ZR, L | ZL,
                                        R | ZR, MINUS | L, MINUS | R, PLUS | L, PLUS | R};

    // Los toques
    boolean touchOn = true;     // los controles en pantalla (apagados: solo el botón del menú)
    int opacity = 70;           // %
    int size = 100;             // %
    int vibration = 2;          // 0 apagada, 1 suave, 2 media, 3 fuerte
    boolean slide = true;       // un dedo que pasa de un botón a otro suelta el primero y pulsa el segundo
    int touchDeadzone = 8;      // % del radio del stick táctil
    boolean stickPress = true;  // un toque corto en un stick es L3 / R3
    boolean cameraSwipe = true; // dragging on the game (away from the controls) turns the camera
    int swipeSpeed = 100;       // %: how far the camera turns for a drag
    final float[][] place = new float[Ctl.values().length][3];
    final boolean[] shown = new boolean[Ctl.values().length];
    final int[] shape = new int[Ctl.values().length];  // SHAPES
    final int[] icon = new int[Ctl.values().length];   // ICONS
    final boolean[] hold = new boolean[Ctl.values().length];  // tap to hold: a tap presses it, the next tap lets go
    final int[] alpha = new int[Ctl.values().length];   // % of the overall opacity (100: as the others)
    final int[] color = new int[Ctl.values().length];   // COLORS
    final String[] label = new String[Ctl.values().length];  // its own text instead of its letters (null: none)
    final boolean[] image = new boolean[Ctl.values().length];  // a picture of the player's (imageFile) instead of the drawing
    final int[] outline = new int[Ctl.values().length];  // the edge's width, % of its usual (0: none)
    final boolean[] fit = new boolean[Ctl.values().length];  // the picture in its own shape (and touched there), not cut to the button's
    private final java.io.File images;  // where those pictures are kept (the app's own files: they stay)
    int imageVersion;  // goes up when a picture changes (TouchPad loads them again)
    boolean grid = false;  // the layout editor's grid: shown, and what's dragged snaps to it
    // Los combos
    final long[] combo = new long[COMBOS];     // los botones que pulsa (0: ninguno, y no sale en pantalla)
    // Button mapping: the controller key (an Android keycode) that presses each of BUTTONS instead of its own
    // (0: the usual one). Analog triggers count as L2 / R2 and a hat as the d-pad keys (android_main.cpp).
    final int[] mapKey = new int[BUTTONS.length];
    final long[] comboPad = new long[COMBOS];  // el botón (o los dos) del mando que lo dispara (0: ninguno)
    // A controller or keyboard key of any kind that fires it too (an Android keycode; 0: none), as the button mapping
    final int[] comboKey = new int[COMBOS];
    // El mando físico
    int padDeadzone = 12;   // % del recorrido del stick
    int trigger = 35;       // % del recorrido de un gatillo analógico desde el que cuenta como ZL / ZR
    int rightAxes = 0;      // 0 según el mando, 1 Z/RZ, 2 RX/RY (android_main.cpp: kAxes*)

    private final SharedPreferences prefs;

    Controls(Context context) {
        prefs = context.getSharedPreferences("controles", Context.MODE_PRIVATE);
        images = new java.io.File(context.getFilesDir(), "touch_images");
        resetLayout();
        load();
    }

    // Everything shows except the right stick: camera swipe turns the camera instead (Show / hide brings it back)
    static boolean shownByDefault(Ctl c) { return c != Ctl.RSTICK; }

    void resetLayout() {
        for (Ctl c : Ctl.values()) {
            place[c.ordinal()] = HOME[c.ordinal()].clone();
            shown[c.ordinal()] = shownByDefault(c);
            shape[c.ordinal()] = icon[c.ordinal()] = color[c.ordinal()] = 0;
            hold[c.ordinal()] = false;
            alpha[c.ordinal()] = 100;
            outline[c.ordinal()] = 100;
            fit[c.ordinal()] = false;
            label[c.ordinal()] = null;  // (pictures stay: Remove takes one away)
        }
    }

    // An added button's name: "Button n" for one of a single button, "Combo n" for several, each counted on its own in
    // the combos' order (an empty one: "Combo n" by its place)
    String comboName(int i) {
        if (combo[i] == 0) return "Empty " + (i + 1);
        boolean single = Long.bitCount(combo[i]) == 1;
        int n = 0;
        for (int k = 0; k <= i; k++)
            if (combo[k] != 0 && (Long.bitCount(combo[k]) == 1) == single) n++;
        return (single ? "Button " : "Combo ") + n;
    }

    // A control's name in the menus (combos by comboName)
    String name(Ctl c) { return c.combo() ? comboName(c.comboIndex()) : label(c); }

    // A control's name in the menus
    static String label(Ctl c) {
        switch (c) {
            case LSTICK: return "Left stick";
            case RSTICK: return "Right stick";
            case DPAD: return "D-pad";
            case MENU: return "Menu";
            case MOD: return "MOD UI";
            case MINUS: return "−";
            case PLUS: return "+";
            default:
                return c.name();
        }
    }

    private void load() {
        touchOn = prefs.getBoolean("tactil", touchOn);
        opacity = prefs.getInt("opacidad", opacity);
        size = prefs.getInt("tamano", size);
        vibration = prefs.getInt("vibracion", vibration);
        slide = prefs.getBoolean("deslizar", slide);
        touchDeadzone = prefs.getInt("zona_tactil", touchDeadzone);
        stickPress = prefs.getBoolean("toque_stick", stickPress);
        cameraSwipe = prefs.getBoolean("camera_swipe", cameraSwipe);
        swipeSpeed = prefs.getInt("swipe_speed", swipeSpeed);
        padDeadzone = prefs.getInt("zona_mando", padDeadzone);
        trigger = prefs.getInt("gatillo", trigger);
        rightAxes = prefs.getInt("stick_derecho", rightAxes);
        for (int i = 0; i < BUTTONS.length; i++) mapKey[i] = prefs.getInt("map_key" + i, 0);
        for (int i = 0; i < COMBOS; i++) {
            combo[i] = prefs.getLong("combo" + i, 0);
            comboPad[i] = prefs.getLong("combo_mando" + i, 0);
            comboKey[i] = prefs.getInt("combo_key" + i, 0);
        }
        grid = prefs.getBoolean("grid", grid);
        // "LSTICK:0.12,0.46,1,1,0,0,0;..." (x, y, tamaño, se ve, forma, icono, tap to hold): lo que no esté, como viene
        for (String item : prefs.getString("disposicion", "").split(";")) {
            String[] kv = item.split(":");
            if (kv.length != 2) continue;
            try {
                Ctl c = Ctl.valueOf(kv[0]);
                String[] v = kv[1].split(",");
                place[c.ordinal()][0] = Float.parseFloat(v[0]);
                place[c.ordinal()][1] = Float.parseFloat(v[1]);
                place[c.ordinal()][2] = Float.parseFloat(v[2]);
                shown[c.ordinal()] = v.length < 4 || !v[3].equals("0");
                if (v.length >= 7) {  // (look and tap to hold: layouts from before them have neither)
                    shape[c.ordinal()] = Math.max(0, Math.min(SHAPES.length - 1, Integer.parseInt(v[4])));
                    icon[c.ordinal()] = Math.max(0, Math.min(ICONS.length - 1, Integer.parseInt(v[5])));
                    hold[c.ordinal()] = v[6].equals("1");
                }
                if (v.length >= 9) {
                    alpha[c.ordinal()] = Math.max(10, Math.min(100, Integer.parseInt(v[7])));
                    color[c.ordinal()] = Math.max(0, Math.min(COLORS.length - 1, Integer.parseInt(v[8])));
                }
                if (v.length >= 11) {
                    outline[c.ordinal()] = Math.max(0, Math.min(300, Integer.parseInt(v[9])));
                    fit[c.ordinal()] = v[10].equals("1");
                }
            } catch (RuntimeException ignored) {
                // (un control que ya no existe, o un número roto: queda el de siempre)
            }
        }
        for (Ctl c : Ctl.values()) {
            label[c.ordinal()] = prefs.getString("label_" + c.name(), null);
            image[c.ordinal()] = new java.io.File(images, c.name() + ".png").isFile();
        }
        // Layouts saved before camera swipe had the right stick on: hide it once, like a new install
        if (prefs.getInt("layout_version", 1) < 2) shown[Ctl.RSTICK.ordinal()] = false;
    }

    void save() {
        edit().apply();
    }

    // Written before returning (the app is going to the background, or the editor closed)
    void saveNow() {
        edit().commit();
    }

    private SharedPreferences.Editor edit() {
        StringBuilder layout = new StringBuilder();
        for (Ctl c : Ctl.values()) {
            float[] p = place[c.ordinal()];
            layout.append(c.name()).append(':').append(p[0]).append(',').append(p[1]).append(',').append(p[2]).append(',')
                    .append(shown[c.ordinal()] ? '1' : '0').append(',').append(shape[c.ordinal()]).append(',')
                    .append(icon[c.ordinal()]).append(',').append(hold[c.ordinal()] ? '1' : '0').append(',')
                    .append(alpha[c.ordinal()]).append(',').append(color[c.ordinal()]).append(',')
                    .append(outline[c.ordinal()]).append(',').append(fit[c.ordinal()] ? '1' : '0').append(';');
        }
        SharedPreferences.Editor e = prefs.edit();
        e.putBoolean("tactil", touchOn).putInt("opacidad", opacity).putInt("tamano", size).putInt("vibracion", vibration)
                .putBoolean("deslizar", slide).putInt("zona_tactil", touchDeadzone).putBoolean("toque_stick", stickPress)
                .putInt("zona_mando", padDeadzone).putInt("gatillo", trigger).putInt("stick_derecho", rightAxes)
                .putBoolean("camera_swipe", cameraSwipe).putInt("swipe_speed", swipeSpeed).putBoolean("grid", grid)
                .putString("disposicion", layout.toString()).putInt("layout_version", 2);
        for (int i = 0; i < COMBOS; i++)
            e.putLong("combo" + i, combo[i]).putLong("combo_mando" + i, comboPad[i]).putInt("combo_key" + i, comboKey[i]);
        for (Ctl c : Ctl.values()) {
            String l = label[c.ordinal()];
            if (l == null || l.isEmpty()) e.remove("label_" + c.name());
            else e.putString("label_" + c.name(), l);
        }
        for (int i = 0; i < BUTTONS.length; i++) e.putInt("map_key" + i, mapKey[i]);
        return e;
    }

    // A control's picture (a PNG the app made from the player's image, at most 256 pixels a side)
    java.io.File imageFile(Ctl c) { return new java.io.File(images, c.name() + ".png"); }

    void setImage(Ctl c, android.graphics.Bitmap bitmap) throws java.io.IOException {
        images.mkdirs();
        java.io.File tmp = new java.io.File(images, c.name() + ".png.new");
        try (java.io.FileOutputStream out = new java.io.FileOutputStream(tmp)) {
            if (!bitmap.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, out)) throw new java.io.IOException("PNG");
        }
        if (!tmp.renameTo(imageFile(c))) throw new java.io.IOException("rename");
        image[c.ordinal()] = true;
        imageVersion++;
    }

    void clearImage(Ctl c) {
        imageFile(c).delete();
        image[c.ordinal()] = false;
        imageVersion++;
    }

    // ---- export / import (a .zip: layout.json and the pictures) ---------------------------------------------------------

    // Everything about the touch controls: what shows, where, how it looks, the combos and their controller buttons
    org.json.JSONObject toJson() throws org.json.JSONException {
        org.json.JSONObject o = new org.json.JSONObject();
        o.put("format", "mhrise-touch-layout").put("version", 1);
        o.put("opacity", opacity).put("size", size).put("slide", slide).put("stickPress", stickPress)
                .put("cameraSwipe", cameraSwipe).put("swipeSpeed", swipeSpeed).put("touchDeadzone", touchDeadzone);
        org.json.JSONObject controls = new org.json.JSONObject();
        for (Ctl c : Ctl.values()) {
            int k = c.ordinal();
            org.json.JSONObject e = new org.json.JSONObject();
            e.put("x", place[k][0]).put("y", place[k][1]).put("scale", place[k][2]).put("shown", shown[k])
                    .put("shape", shape[k]).put("icon", icon[k]).put("hold", hold[k]).put("alpha", alpha[k])
                    .put("color", color[k]).put("image", image[k]).put("outline", outline[k]).put("fit", fit[k]);
            if (label[k] != null) e.put("label", label[k]);
            if (c.combo())
                e.put("buttons", combo[c.comboIndex()]).put("pad", comboPad[c.comboIndex()]).put("key", comboKey[c.comboIndex()]);
            controls.put(c.name(), e);
        }
        o.put("controls", controls);
        return o;
    }

    // After an import put the pictures in place (or took them away)
    void rescanImages() {
        for (Ctl c : Ctl.values()) image[c.ordinal()] = imageFile(c).isFile();
        imageVersion++;
    }

    java.io.File imagesDir() { return images; }

    // What toJson wrote (another phone's too: places are fractions of the screen). Unknown controls are skipped and
    // missing ones keep their place; the pictures are put in place by the caller.
    void fromJson(org.json.JSONObject o) throws org.json.JSONException {
        if (!"mhrise-touch-layout".equals(o.optString("format"))) throw new org.json.JSONException("not a layout");
        opacity = clampInt(o.optInt("opacity", opacity), 20, 100);
        size = clampInt(o.optInt("size", size), 60, 160);
        slide = o.optBoolean("slide", slide);
        stickPress = o.optBoolean("stickPress", stickPress);
        cameraSwipe = o.optBoolean("cameraSwipe", cameraSwipe);
        swipeSpeed = clampInt(o.optInt("swipeSpeed", swipeSpeed), 25, 300);
        touchDeadzone = clampInt(o.optInt("touchDeadzone", touchDeadzone), 0, 50);
        org.json.JSONObject controls = o.getJSONObject("controls");
        for (Ctl c : Ctl.values()) {
            org.json.JSONObject e = controls.optJSONObject(c.name());
            if (e == null) continue;
            int k = c.ordinal();
            place[k][0] = (float) Math.max(0.02, Math.min(0.98, e.optDouble("x", place[k][0])));
            place[k][1] = (float) Math.max(0.03, Math.min(0.97, e.optDouble("y", place[k][1])));
            place[k][2] = (float) Math.max(0.5, Math.min(2.5, e.optDouble("scale", place[k][2])));
            shown[k] = e.optBoolean("shown", shown[k]);
            shape[k] = clampInt(e.optInt("shape", 0), 0, SHAPES.length - 1);
            icon[k] = clampInt(e.optInt("icon", 0), 0, ICONS.length - 1);
            hold[k] = e.optBoolean("hold", false);
            alpha[k] = clampInt(e.optInt("alpha", 100), 10, 100);
            color[k] = clampInt(e.optInt("color", 0), 0, COLORS.length - 1);
            label[k] = e.has("label") ? e.getString("label") : null;
            outline[k] = clampInt(e.optInt("outline", 100), 0, 300);
            fit[k] = e.optBoolean("fit", false);
            if (c.combo()) {
                combo[c.comboIndex()] = e.optLong("buttons", 0);
                comboPad[c.comboIndex()] = e.optLong("pad", 0);
                comboKey[c.comboIndex()] = e.optInt("key", 0);
            }
        }
    }

    private static int clampInt(int v, int lo, int hi) { return Math.max(lo, Math.min(hi, v)); }

    // The console buttons a touch button presses (0: a stick, the d-pad, the menu, or an empty combo)
    long buttons(Ctl c) {
        switch (c) {
            case A: return A;
            case B: return B;
            case X: return X;
            case Y: return Y;
            case L: return L;
            case R: return R;
            case ZL: return ZL;
            case ZR: return ZR;
            case MINUS: return MINUS;
            case PLUS: return PLUS;
            case L3: return LSTICK;
            case R3: return RSTICK;
            default: return c.combo() ? combo[c.comboIndex()] : 0;
        }
    }

    // A button that presses something by itself: it can have an icon or a label, tap to hold, a copy (not the sticks,
    // the d-pad or the menu)
    static boolean isButton(Ctl c) {
        return c != null && c != Ctl.LSTICK && c != Ctl.RSTICK && c != Ctl.DPAD && c != Ctl.MENU && c != Ctl.MOD;
    }

    static boolean isStick(Ctl c) { return c == Ctl.LSTICK || c == Ctl.RSTICK; }

    // Whether its look was changed from the usual drawing (Edit layout: size and opacity aside)
    boolean styled(Ctl c) {
        int k = c.ordinal();
        return shape[k] != 0 || icon[k] != 0 || hold[k] || color[k] != 0 || label[k] != null || image[k] || outline[k] != 100;
    }

    // The editor's Delete: an added button goes, with its controller buttons and its look (its place is free for Add)
    void removeButton(int k) {
        combo[k] = comboPad[k] = 0;
        comboKey[k] = 0;
        int o = Ctl.COMBO1.ordinal() + k;
        place[o] = HOME[o].clone();
        shown[o] = true;
        shape[o] = icon[o] = color[o] = 0;
        hold[o] = fit[o] = false;
        alpha[o] = outline[o] = 100;
        label[o] = null;
        if (image[o]) clearImage(Ctl.values()[o]);
    }

    // The first combo with no buttons (-1: all taken)
    int freeCombo() {
        for (int i = 0; i < COMBOS; i++)
            if (combo[i] == 0) return i;
        return -1;
    }

    // Los combos del mando para el loader: pares (botones del mando, botones que pulsa), los de dos botones primero
    long[] padCombos() {
        long[] out = new long[COMBOS * 2];
        int n = 0;
        for (int pass = 0; pass < 2; pass++)
            for (int i = 0; i < COMBOS; i++) {
                if (comboPad[i] == 0 || combo[i] == 0) continue;
                boolean two = Long.bitCount(comboPad[i]) > 1;
                if ((pass == 0) != two) continue;
                out[n++] = comboPad[i];
                out[n++] = combo[i];
            }
        return java.util.Arrays.copyOf(out, n);
    }

    // The button mapping for the loader: pairs (keycode, the buttons it presses)
    long[] padMapping() {
        long[] out = new long[(BUTTONS.length + COMBOS) * 2];
        int n = 0;
        for (int i = 0; i < BUTTONS.length; i++) {
            if (mapKey[i] == 0) continue;
            out[n++] = mapKey[i];
            out[n++] = BUTTONS[i];
        }
        for (int i = 0; i < COMBOS; i++) {  // (an added button's or a combo's key: it presses all of its buttons)
            if (comboKey[i] == 0 || combo[i] == 0) continue;
            out[n++] = comboKey[i];
            out[n++] = combo[i];
        }
        return java.util.Arrays.copyOf(out, n);
    }

    // A controller key's name in the menu ("A", "L1", "Start", "D-pad up"…)
    static String keyName(int keycode) {
        switch (keycode) {
            case android.view.KeyEvent.KEYCODE_DPAD_UP: return "D-pad up";
            case android.view.KeyEvent.KEYCODE_DPAD_DOWN: return "D-pad down";
            case android.view.KeyEvent.KEYCODE_DPAD_LEFT: return "D-pad left";
            case android.view.KeyEvent.KEYCODE_DPAD_RIGHT: return "D-pad right";
            case android.view.KeyEvent.KEYCODE_BUTTON_THUMBL: return "Left stick press";
            case android.view.KeyEvent.KEYCODE_BUTTON_THUMBR: return "Right stick press";
            case android.view.KeyEvent.KEYCODE_BACK: return "Back";
            default:
        }
        String s = android.view.KeyEvent.keyCodeToString(keycode);
        if (s.startsWith("KEYCODE_")) s = s.substring(8);
        if (s.startsWith("BUTTON_")) s = s.substring(7);
        if (s.length() <= 2) return s;
        return s.charAt(0) + s.substring(1).toLowerCase(java.util.Locale.ROOT).replace('_', ' ');
    }
}
