package io.github.iamwilmar.mhriseport;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.RenderNode;
import android.os.SystemClock;
import android.view.Choreographer;
import android.view.MotionEvent;
import android.view.View;

import io.github.iamwilmar.mhriseport.Controls.Ctl;

// Los controles táctiles, en una ventana transparente encima del juego (GameActivity): la disposición tradicional de un
// mando con todo (los dos sticks, la cruceta, A B X Y, L R ZL ZR, − +, L3 y R3) y los combos que haya. Cada dedo es de
// un control: un botón se pulsa mientras el dedo esté encima, un stick sigue al dedo desde su centro, y un toque corto en
// un stick (sin moverlo) es L3 / R3. Con "deslizar" un dedo que pasa a otro botón suelta el primero y pulsa el segundo.
// Lo que no es de ningún control va al panel del HUD (nativeTouchPoint), como si no hubiera controles.
// En modo edición cada control se arrastra a su sitio; la barra de arriba cambia el tamaño del elegido, lo esconde o lo
// enseña, y lo devuelve todo a su sitio.
@SuppressLint("ViewConstructor")
final class TouchPad extends View {
    interface Host {
        void send(long buttons, int sticksOn, float lx, float ly, float rx, float ry);
        void touchPoint(int action, float x, float y, long timeNs, int width, int height);
        boolean modTouch(int action, int id, float screenX, float screenY);
        void tick();       // la vibración de un botón pulsado
        void openMenu();
        void editDone();
        // A control was moved, resized, hidden or reset while editing: saved right away (leaving the editor any other
        // way than Done, or the app being closed, would lose it otherwise)
        void layoutChanged();
        // The editor's Edit: a button's size, shape, icon and tap to hold (and a combo's buttons)
        void editControl(Ctl c);
        // The editor's Add: a new button (combo k, already placed in the middle); its buttons are chosen first
        void addButton(int k);
        // The MOD control: the MOD UI opens or closes
        default void toggleModUi() {}
        // The editor's Delete took an added button away (its controller buttons too)
        default void buttonRemoved() { layoutChanged(); }
    }

    // The MOD control is only there with Lua scripts installed (GameActivity: setModsAvailable)
    private boolean modsAvailable;

    void setModsAvailable(boolean available) {
        if (modsAvailable == available) return;
        modsAvailable = available;
        invalidate();
    }

    private final Controls cfg;
    private final Host host;
    private static final Ctl[] CONTROLS = Ctl.values();
    private final boolean animateControls;
    private final Appearance[] artwork = new Appearance[CONTROLS.length];
    private long dirtyGlyphs = -1L;
    private final Rect damage = new Rect();
    // A moving stick redraws the controls at the game's own frame rate (30, or 60 with the 60 fps switch: setGameFps),
    // not at every touch event or vsync. Redrawn at the screen's 90-144 Hz, the overlay kept the GPU (or, drawn by the
    // CPU, a core) busy for frames the game never shows: playing with the touch controls was slower than with a pad.
    private long animationFrameNs = 1_000_000_000L / 60;
    private final Choreographer.FrameCallback animationFrame = this::drawAnimatedControls;
    private boolean animationFramePending;
    private long nextAnimationFrame;

    void setGameFps(int fps) {
        animationFrameNs = 1_000_000_000L / Math.max(20, Math.min(60, fps));
    }

    // Per-control visual state. Layout and style changes invalidate all display lists.
    private static final class Appearance {
        boolean down, held, click;
        int dirs;
        float x, y;
        RenderNode node;

        boolean set(boolean down, boolean held, boolean click, int dirs, float x, float y) {
            if (this.down == down && this.held == held && this.click == click && this.dirs == dirs
                    && this.x == x && this.y == y) return false;
            this.down = down;
            this.held = held;
            this.click = click;
            this.dirs = dirs;
            this.x = x;
            this.y = y;
            return true;
        }
    }
    private final Glyphs glyphs = new Glyphs();
    private final Paint barPaint = new Paint(Paint.ANTI_ALIAS_FLAG), barText = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final RectF rect = new RectF();
    // The player's pictures for the controls (Controls.image), loaded once per imageVersion
    private final android.graphics.Bitmap[] bitmaps = new android.graphics.Bitmap[Ctl.values().length];
    private int bitmapsVersion = -1;

    private android.graphics.Bitmap bitmap(Ctl c) {
        if (bitmapsVersion != cfg.imageVersion) {
            java.util.Arrays.fill(bitmaps, null);
            java.util.Arrays.fill(shapes, null);
            for (Ctl x : Ctl.values())
                if (cfg.image[x.ordinal()]) {
                    bitmaps[x.ordinal()] = android.graphics.BitmapFactory.decodeFile(cfg.imageFile(x).getPath());
                    if (bitmaps[x.ordinal()] != null) shapes[x.ordinal()] = Outline.of(bitmaps[x.ordinal()]);
                }
            bitmapsVersion = cfg.imageVersion;
        }
        return bitmaps[c.ordinal()];
    }

    // A picture's own shape, for touches: in each row, the first and the last pixel that isn't (nearly) see-through.
    // Between them counts as inside, so holes in the picture still take the finger. A picture with too little of it
    // solid (a thin outline, a faint mark) isn't used as a shape: its square's inscribed circle is, as for others.
    private static final class Outline {
        int[] left, right;
        boolean usable;

        static Outline of(android.graphics.Bitmap b) {
            Outline o = new Outline();
            int w = b.getWidth(), h = b.getHeight();
            o.left = new int[h];
            o.right = new int[h];
            int[] row = new int[w];
            long solid = 0;
            for (int y = 0; y < h; y++) {
                b.getPixels(row, 0, w, 0, y, w, 1);
                int l = -1, r = -1;
                for (int x = 0; x < w; x++)
                    if ((row[x] >>> 24) >= 40) {
                        if (l < 0) l = x;
                        r = x;
                    }
                o.left[y] = l;
                o.right[y] = r;
                if (l >= 0) solid += r - l + 1;
            }
            o.usable = solid >= (long) w * h * 25 / 100;
            return o;
        }
    }
    private final Outline[] shapes = new Outline[Ctl.values().length];

    // For a picture in its own shape: the touch inside it (true), outside (false), or no shape to go by (null)
    private Boolean insidePicture(Ctl c, float x, float y) {
        // (a stick's picture is its knob, which moves: the stick is touched on its base, as always)
        if (!cfg.image[c.ordinal()] || !cfg.fit[c.ordinal()] || Controls.isStick(c)) return null;
        android.graphics.Bitmap b = bitmap(c);
        Outline o = shapes[c.ordinal()];
        if (b == null || o == null || !o.usable) return null;
        glyphs.fitted(b, cx(c), cy(c), radius(c));
        RectF box = glyphs.fittedBox();
        float pad = radius(c) * 0.12f;  // (a little past its edge: a thumb isn't a pixel)
        if (x < box.left - pad || x > box.right + pad || y < box.top - pad || y > box.bottom + pad) return false;
        int py = Math.max(0, Math.min(b.getHeight() - 1, (int) ((y - box.top) / box.height() * b.getHeight())));
        int px = (int) ((x - box.left) / box.width() * b.getWidth());
        int margin = Math.max(1, (int) (pad / box.width() * b.getWidth()));
        // (the row itself and its neighbours, so a pointed top or bottom isn't missed)
        for (int dy = -margin; dy <= margin; dy++) {
            int yy = py + dy;
            if (yy < 0 || yy >= b.getHeight() || o.left[yy] < 0) continue;
            if (px >= o.left[yy] - margin && px <= o.right[yy] + margin) return true;
        }
        return false;
    }

    private static final int MAX_FINGERS = 10;
    private static final class Finger {
        int id = -1;
        Ctl ctl;          // null: no es de ningún control (va al HUD, si es el primero)
        boolean hud, modWindow;
        float x0, y0, x, y;
        long down;
        boolean moved;
        long mask;        // los botones que pulsa ahora
        int dirs;         // la cruceta: los brazos
        float dx, dy;     // en edición: la distancia del dedo al centro del control
        boolean camera;   // a swipe on the game: it moves the right stick
        float lastX, lastY;
        long lastT;
    }
    // Camera swipe: a finger that drags on the game, away from the controls, turns the camera through the right stick, by
    // how fast it moves (a phone game's swipe camera; the game only has a stick to turn it with). A tap still goes to the
    // game's touch screen. The stick goes back to the middle when the finger stops.
    private final Runnable cameraStop = () -> {
        rx = ry = 0;
        update();
    };
    private final Finger[] fingers = new Finger[MAX_FINGERS];
    private long pulse;  // L3 / R3 de un toque corto en un stick, unos instantes
    private final boolean[] latched = new boolean[Ctl.values().length];  // tap to hold: pressed until tapped again
    private float lx, ly, rx, ry;
    private int sticksOn;
    private long lastButtons = -1;
    private int lastSticks = -1;

    boolean editing;
    private Ctl selected;

    TouchPad(Context context, Controls cfg, Host host) {
        this(context, cfg, host, !(context instanceof GameActivity &&
                new Ini(SettingsMenu.gameFolder((GameActivity) context)).on("static_touch_panel")));
    }

    TouchPad(Context context, Controls cfg, Host host, boolean animateControls) {
        super(context);
        this.cfg = cfg;
        this.host = host;
        this.animateControls = animateControls;
        for (int k = 0; k < artwork.length; ++k) artwork[k] = new Appearance();
        for (int i = 0; i < MAX_FINGERS; i++) fingers[i] = new Finger();
        barText.setTextAlign(Paint.Align.CENTER);
        barText.setColor(Glyphs.CREAM);
    }

    // ---- dónde y cómo de grande es cada control ---------------------------------------------------------------------

    private float unit() { return getHeight() * cfg.size / 100f; }
    private float cx(Ctl c) { return cfg.place[c.ordinal()][0] * getWidth(); }
    private float cy(Ctl c) { return cfg.place[c.ordinal()][1] * getHeight(); }
    private float scale(Ctl c) { return cfg.place[c.ordinal()][2]; }

    // El radio de un control (el de la mitad de su ancho para L, R, ZL y ZR)
    private float radius(Ctl c) {
        float u = unit() * scale(c);
        switch (c) {
            case LSTICK: case RSTICK: return 0.15f * u;
            case DPAD: return 0.13f * u;
            case L: case R: return 0.075f * u;
            case ZL: case ZR: return 0.06f * u;
            case MINUS: case PLUS: case MENU: case MOD: case L3: case R3: return 0.042f * u;
            case A: case B: case X: case Y: return 0.062f * u;
            default: return 0.06f * u;  // combos
        }
    }

    private boolean active(Ctl c) {
        if (c == Ctl.MENU) return true;  // (siempre: es la forma de volver a encender los demás)
        if (!cfg.touchOn) return false;
        if (c == Ctl.MOD && !modsAvailable) return false;
        if (c.combo() && cfg.combo[c.comboIndex()] == 0) return false;
        return cfg.shown[c.ordinal()] || editing;
    }

    // El control bajo un punto: el más cercano a su centro entre los que lo tienen dentro (un poco más grandes de lo
    // que se ven, que un pulgar no es fino); null si ninguno
    private Ctl hit(float x, float y) {
        Ctl best = null;
        float bestD = Float.MAX_VALUE;
        for (Ctl c : Ctl.values()) {
            if (!active(c) || (!editing && !cfg.shown[c.ordinal()])) continue;
            float r = radius(c), dx = x - cx(c), dy = y - cy(c);
            final int shape = cfg.shape[c.ordinal()];
            final boolean pill = (c == Ctl.L || c == Ctl.R) && shape == 0;
            float reach = c == Ctl.LSTICK || c == Ctl.RSTICK ? r * 1.35f : pill ? r * 1.15f : r * 1.25f;
            if (pill) dy *= 2.0f;  // (una pastilla: más ancha que alta)
            float d = shape >= 2 && c != Ctl.DPAD ? Math.max(Math.abs(dx), Math.abs(dy))  // (a square)
                                                         : (float) Math.hypot(dx, dy);
            Boolean picture = insidePicture(c, x, y);
            if (picture != null) {
                if (!picture) continue;
                d = Math.min(d, reach * 0.99f);  // (inside its shape: it counts, nearer its middle first)
            }
            if (d <= reach && d / reach < bestD) {
                best = c;
                bestD = d / reach;
            }
        }
        return best;
    }

    private static boolean isButton(Ctl c) {
        return c != null && c != Ctl.LSTICK && c != Ctl.RSTICK && c != Ctl.MENU && c != Ctl.MOD;
    }

    // ---- los dedos ----------------------------------------------------------------------------------------------------

    private Finger finger(int id) {
        for (Finger f : fingers) if (f.id == id) return f;
        return null;
    }
    private Finger free() {
        for (Finger f : fingers) if (f.id < 0) return f;
        return null;
    }
    private boolean hudTaken() {
        for (Finger f : fingers) if (f.id >= 0 && f.hud) return true;
        return false;
    }

    @SuppressLint("ClickableViewAccessibility")
    @Override
    public boolean onTouchEvent(MotionEvent e) {
        int action = e.getActionMasked();
        int index = e.getActionIndex();
        long timeNs = e.getEventTime() * 1_000_000L;
        switch (action) {
            case MotionEvent.ACTION_DOWN:
            case MotionEvent.ACTION_POINTER_DOWN: {
                Finger f = free();
                if (f == null) break;
                f.id = e.getPointerId(index);
                f.x = f.x0 = e.getX(index);
                f.y = f.y0 = e.getY(index);
                f.down = SystemClock.uptimeMillis();
                f.moved = false;
                f.mask = 0;
                f.dirs = 0;
                f.hud = false;
                f.modWindow = !editing && host.modTouch(MotionEvent.ACTION_DOWN, f.id,
                        e.getRawX(index), e.getRawY(index));
                if (f.modWindow) {
                    f.ctl = null;
                    break;
                }
                if (editing && toolbar(f.x, f.y)) {
                    f.ctl = null;
                    break;
                }
                f.ctl = hit(f.x, f.y);
                if (editing) {
                    selected = f.ctl;
                    if (f.ctl != null) {
                        f.dx = cx(f.ctl) - f.x;
                        f.dy = cy(f.ctl) - f.y;
                    }
                } else if (f.ctl == null) {
                    if (!hudTaken()) {  // (el panel del HUD sigue un dedo: el primero que no es de un control)
                        f.hud = true;
                        host.touchPoint(MotionEvent.ACTION_DOWN, f.x, f.y, timeNs, getWidth(), getHeight());
                    }
                } else {
                    press(f, true);
                }
                break;
            }
            case MotionEvent.ACTION_MOVE:
                for (int i = 0; i < e.getPointerCount(); i++) {
                    Finger f = finger(e.getPointerId(i));
                    if (f == null) continue;
                    if (f.modWindow) {
                        host.modTouch(MotionEvent.ACTION_MOVE, f.id, e.getRawX(i), e.getRawY(i));
                        continue;
                    }
                    f.x = e.getX(i);
                    f.y = e.getY(i);
                    if (Math.hypot(f.x - f.x0, f.y - f.y0) > getHeight() * 0.025f) f.moved = true;
                    if (editing) {
                        if (f.ctl != null) {
                            float[] p = cfg.place[f.ctl.ordinal()];
                            float px = f.x + f.dx, py = f.y + f.dy;
                            if (cfg.grid) {  // (the center on the grid's nearest crossing)
                                float g = gridStep();
                                px = Math.round(px / g) * g;
                                py = Math.round(py / g) * g;
                            }
                            p[0] = clamp(px / getWidth(), 0.02f, 0.98f);
                            p[1] = clamp(py / getHeight(), 0.03f, 0.97f);
                        }
                    } else if (f.camera) {
                        swipe(f, e.getEventTime());
                    } else if (f.ctl == null && cfg.cameraSwipe && f.moved) {
                        // (from touch to camera: the game gets its touch cancelled)
                        if (f.hud) host.touchPoint(MotionEvent.ACTION_CANCEL, f.x, f.y, timeNs, getWidth(), getHeight());
                        f.hud = false;
                        f.camera = true;
                        f.lastX = f.x;
                        f.lastY = f.y;
                        f.lastT = e.getEventTime();
                    } else if (f.hud) {
                        host.touchPoint(MotionEvent.ACTION_MOVE, f.x, f.y, timeNs, getWidth(), getHeight());
                    } else if (f.ctl != null) {
                        move(f);
                    }
                }
                break;
            case MotionEvent.ACTION_UP:
            case MotionEvent.ACTION_POINTER_UP:
            case MotionEvent.ACTION_CANCEL: {
                boolean all = action == MotionEvent.ACTION_CANCEL;
                if (editing) host.layoutChanged();
                for (int i = 0; i < e.getPointerCount(); i++) {
                    if (!all && i != index) continue;
                    Finger f = finger(e.getPointerId(i));
                    if (f == null) continue;
                    if (f.modWindow) {
                        host.modTouch(all ? MotionEvent.ACTION_CANCEL : MotionEvent.ACTION_UP, f.id,
                                e.getRawX(i), e.getRawY(i));
                        f.modWindow = false;
                    } else if (f.hud) {
                        host.touchPoint(all ? MotionEvent.ACTION_CANCEL : MotionEvent.ACTION_UP, e.getX(i), e.getY(i), timeNs,
                                getWidth(), getHeight());
                    } else if (!editing && f.ctl != null) {
                        release(f, !all);
                    }
                    if (f.camera) {
                        removeCallbacks(cameraStop);
                        rx = ry = 0;
                        f.camera = false;
                    }
                    f.id = -1;
                    f.ctl = null;
                    f.hud = false;
                }
                break;
            }
            default:
                break;
        }
        update(action == MotionEvent.ACTION_MOVE);
        return true;
    }

    private static float clamp(float v, float lo, float hi) { return Math.max(lo, Math.min(hi, v)); }

    // Full stick for a drag of about half the screen's height a second (at 100 %), smoothed over a few events
    private void swipe(Finger f, long timeMs) {
        float dt = Math.max(timeMs - f.lastT, 4) / 1000f;
        float gain = 2.0f * cfg.swipeSpeed / 100f / getHeight();
        float vx = clamp((f.x - f.lastX) / dt * gain, -1, 1), vy = clamp((f.y - f.lastY) / dt * gain, -1, 1);
        rx = rx * 0.4f + vx * 0.6f;
        ry = ry * 0.4f + vy * 0.6f;
        f.lastX = f.x;
        f.lastY = f.y;
        f.lastT = timeMs;
        removeCallbacks(cameraStop);
        postDelayed(cameraStop, 70);
    }

    private void press(Finger f, boolean first) {
        Ctl c = f.ctl;
        if (c == Ctl.LSTICK || c == Ctl.RSTICK) {
            stickTo(f);
            return;
        }
        if (c == Ctl.DPAD) {
            int was = f.dirs;
            f.dirs = dpadDirs(f);
            f.mask = dirMask(f.dirs);
            if (f.dirs != 0 && f.dirs != was) host.tick();
            return;
        }
        if (c == Ctl.MENU || c == Ctl.MOD) return;  // (al soltarlo)
        if (cfg.hold[c.ordinal()]) {  // (tap to hold: each new touch flips it; the finger itself presses nothing)
            f.mask = 0;
            if (first) {
                latched[c.ordinal()] = !latched[c.ordinal()];
                host.tick();
            }
            return;
        }
        f.mask = cfg.buttons(c);
        if (first || f.mask != 0) host.tick();
    }

    private void move(Finger f) {
        Ctl c = f.ctl;
        if (c == Ctl.LSTICK || c == Ctl.RSTICK) {
            stickTo(f);
            return;
        }
        if (c == Ctl.DPAD) {
            Ctl under = cfg.slide ? hit(f.x, f.y) : Ctl.DPAD;
            if (under == Ctl.DPAD || under == null || !isButton(under)) {
                press(f, false);
                return;
            }
        }
        if (!cfg.slide) return;
        Ctl under = hit(f.x, f.y);
        if (under != null && under != c && isButton(under)) {  // (al pasar a otro botón: suelta este y pulsa aquel)
            f.ctl = under;
            f.dirs = 0;
            press(f, true);
        }
    }

    private void release(Finger f, boolean lifted) {
        Ctl c = f.ctl;
        if (c == Ctl.MENU && lifted && !f.moved) host.openMenu();
        if (c == Ctl.MOD && lifted && !f.moved) {
            host.tick();
            host.toggleModUi();
        }
        if ((c == Ctl.LSTICK || c == Ctl.RSTICK) && lifted && cfg.stickPress) {
            // un toque corto sin mover el stick: L3 / R3
            long held = SystemClock.uptimeMillis() - f.down;
            if (held < 250 && Math.hypot(f.x - f.x0, f.y - f.y0) < radius(c) * 0.18f) {
                final long bit = c == Ctl.LSTICK ? Controls.LSTICK : Controls.RSTICK;
                pulse |= bit;
                host.tick();
                postDelayed(() -> {
                    pulse &= ~bit;
                    update();
                }, 90);
            }
        }
        if (c == Ctl.LSTICK) lx = ly = 0;
        if (c == Ctl.RSTICK) rx = ry = 0;
        f.mask = 0;
        f.dirs = 0;
    }

    // El stick sigue al dedo desde su centro; fuera del aro, al tope en esa dirección. La zona muerta del menú, radial.
    private void stickTo(Finger f) {
        Ctl c = f.ctl;
        float r = radius(c) * 0.8f;
        float x = (f.x - cx(c)) / r, y = (f.y - cy(c)) / r;
        float m = (float) Math.hypot(x, y);
        if (m > 1) {
            x /= m;
            y /= m;
            m = 1;
        }
        float zone = cfg.touchDeadzone / 100f;
        if (m <= zone) {
            x = y = 0;
        } else {
            float k = (m - zone) / (1 - zone) / m;
            x *= k;
            y *= k;
        }
        if (c == Ctl.LSTICK) {
            lx = x;
            ly = y;
        } else {
            rx = x;
            ry = y;
        }
    }

    // La cruceta en ocho sectores: arriba, abajo, izquierda, derecha y las diagonales (dos brazos a la vez)
    private int dpadDirs(Finger f) {
        float dx = f.x - cx(Ctl.DPAD), dy = f.y - cy(Ctl.DPAD);
        if (Math.hypot(dx, dy) < radius(Ctl.DPAD) * 0.22f) return 0;
        double a = Math.toDegrees(Math.atan2(dy, dx));  // 0 a la derecha, 90 abajo
        int sector = (int) Math.floor(((a + 360 + 22.5) % 360) / 45);
        switch (sector) {
            case 0: return 8;       // derecha
            case 1: return 8 | 2;   // abajo a la derecha
            case 2: return 2;       // abajo
            case 3: return 4 | 2;
            case 4: return 4;       // izquierda
            case 5: return 4 | 1;
            case 6: return 1;       // arriba
            default: return 8 | 1;
        }
    }
    private static long dirMask(int dirs) {
        long m = 0;
        if ((dirs & 1) != 0) m |= Controls.UP;
        if ((dirs & 2) != 0) m |= Controls.DOWN;
        if ((dirs & 4) != 0) m |= Controls.LEFT;
        if ((dirs & 8) != 0) m |= Controls.RIGHT;
        return m;
    }

    // Lo que tocan los dedos, al loader (solo si cambió)
    private void update() {
        update(false);
    }

    private void update(boolean moving) {
        long buttons = pulse;
        if (!editing)
            for (Ctl c : Ctl.values())
                if (latched[c.ordinal()] && cfg.hold[c.ordinal()] && active(c)) buttons |= cfg.buttons(c);
        int on = 0;
        for (Finger f : fingers) {
            if (f.id < 0 || f.ctl == null || editing) continue;
            buttons |= f.mask;
            if (f.ctl == Ctl.LSTICK) on |= 1;
            if (f.ctl == Ctl.RSTICK) on |= 2;
        }
        for (Finger f : fingers)
            if (f.id >= 0 && f.camera && !editing) on |= 2;
        if (on == 0 && (rx != 0 || ry != 0)) on |= 2;  // (the last swipe's stick going back to the middle)
        sticksOn = on;
        if (buttons != lastButtons || on != 0 || lastSticks != on) {
            host.send(buttons, on, lx, -ly, rx, -ry);  // (al juego, hacia arriba positivo)
            lastButtons = buttons;
            lastSticks = on;
        }
        if (editing) invalidate();
        else if (moving && animateControls) {
            if (!animationFramePending) {
                animationFramePending = true;
                Choreographer.getInstance().postFrameCallback(animationFrame);
            }
        } else if (animateControls && syncAppearance(isHardwareAccelerated() ? null : damage)) {
            if (isHardwareAccelerated()) super.invalidate();
            else super.invalidate(damage);
        }
    }

    private void drawAnimatedControls(long frameTime) {
        if (frameTime + 500_000L < nextAnimationFrame) {
            Choreographer.getInstance().postFrameCallback(animationFrame);
            return;
        }
        animationFramePending = false;
        nextAnimationFrame = frameTime >= nextAnimationFrame + animationFrameNs
                ? frameTime + animationFrameNs : nextAnimationFrame + animationFrameNs;
        if (editing) return;
        if (isHardwareAccelerated()) {
            if (syncAppearance(null)) super.invalidate();
        } else if (syncAppearance(damage)) super.invalidate(damage);
    }

    private boolean syncAppearance(Rect changedArea) {
        boolean changed = false;
        if (changedArea != null) changedArea.setEmpty();
        boolean live = animateControls && !editing;
        for (Ctl ctl : CONTROLS) {
            if (!active(ctl)) continue;
            int k = ctl.ordinal();
            boolean left = ctl == Ctl.LSTICK, right = ctl == Ctl.RSTICK;
            long clickBit = left || ctl == Ctl.L3 ? Controls.LSTICK : right || ctl == Ctl.R3 ? Controls.RSTICK : 0;
            boolean down = live && !left && !right && ctl != Ctl.DPAD &&
                    (pressed(ctl) || (latched[k] && cfg.hold[k]));
            boolean held = live && ((left && (sticksOn & 1) != 0) || (right && (sticksOn & 2) != 0));
            int dirs = live && ctl == Ctl.DPAD ? dpadHeld() : 0;
            float x = live && left ? lx : live && right ? rx : 0;
            float y = live && left ? ly : live && right ? ry : 0;
            if (artwork[k].set(down, held, live && (pulse & clickBit) != 0, dirs, x, y)) {
                dirtyGlyphs |= 1L << k;
                changed = true;
                if (changedArea != null) {
                    // Custom text and combo names can extend past the button plate.
                    if (cfg.label[k] != null || ctl.combo()) changedArea.set(0, 0, getWidth(), getHeight());
                    else {
                        // Includes the stick knob at full travel, its widest outline and antialiasing.
                        float extent = 1.25f * radius(ctl) + 2;
                        changedArea.union((int) Math.floor(cx(ctl) - extent), (int) Math.floor(cy(ctl) - extent),
                                (int) Math.ceil(cx(ctl) + extent), (int) Math.ceil(cy(ctl) + extent));
                    }
                }
            }
        }
        return changed;
    }

    @Override public void invalidate() {
        dirtyGlyphs = -1L;
        super.invalidate();
    }

    @Override protected void onSizeChanged(int w, int h, int oldW, int oldH) {
        super.onSizeChanged(w, h, oldW, oldH);
        dirtyGlyphs = -1L;
    }

    @Override protected void onDetachedFromWindow() {
        Choreographer.getInstance().removeFrameCallback(animationFrame);
        animationFramePending = false;
        nextAnimationFrame = 0;
        for (Appearance a : artwork) if (a.node != null) a.node.discardDisplayList();
        dirtyGlyphs = -1L;
        super.onDetachedFromWindow();
    }

    // Suelta todo (al abrir el menú o al irse la app a segundo plano)
    void releaseAll() {
        for (Finger f : fingers) {
            if (f.modWindow) {
                host.modTouch(MotionEvent.ACTION_CANCEL, f.id, 0, 0);
                f.modWindow = false;
            }
            if (f.hud) host.touchPoint(MotionEvent.ACTION_CANCEL, f.x, f.y, SystemClock.uptimeMillis() * 1_000_000L, getWidth(),
                    getHeight());
            f.id = -1;
            f.ctl = null;
            f.hud = false;
            f.camera = false;
            f.mask = 0;
        }
        removeCallbacks(cameraStop);
        pulse = 0;
        java.util.Arrays.fill(latched, false);
        lx = ly = rx = ry = 0;
        update();
    }

    // ---- la barra del modo edición ------------------------------------------------------------------------------------

    private static final int BAR_SMALLER = 0, BAR_BIGGER = 1, BAR_EDIT = 2, BAR_ADD = 3, BAR_DELETE = 4, BAR_HIDE = 5,
            BAR_GRID = 6, BAR_RESET = 7, BAR_DONE = 8, BAR_ITEMS = 9;

    // The grid's step: a fortieth of the screen's height
    private float gridStep() { return getHeight() / 40f; }

    private RectF barItem(int i) {
        float h = getHeight() * 0.085f, w = getHeight() * 0.17f, gap = h * 0.3f;
        float total = BAR_ITEMS * w + (BAR_ITEMS - 1) * gap;
        float x = (getWidth() - total) / 2 + i * (w + gap), y = getHeight() * 0.18f;
        return new RectF(x, y, x + w, y + h);
    }
    private String barLabel(int i) {
        switch (i) {
            case BAR_SMALLER: return "−";
            case BAR_BIGGER: return "+";
            case BAR_EDIT: return "Edit";
            case BAR_ADD: return "Add";
            case BAR_DELETE: return "Delete";
            case BAR_HIDE: return selected != null && !cfg.shown[selected.ordinal()] ? "Show" : "Hide";
            case BAR_GRID: return cfg.grid ? "Grid ✓" : "Grid";
            case BAR_RESET: return "Reset";
            default: return "Done";
        }
    }

    // Un toque en la barra: true si era suyo
    private boolean toolbar(float x, float y) {
        for (int i = 0; i < BAR_ITEMS; i++) {
            if (!barItem(i).contains(x, y)) continue;
            switch (i) {
                case BAR_SMALLER:
                case BAR_BIGGER:
                    if (selected != null) {
                        float[] p = cfg.place[selected.ordinal()];
                        p[2] = clamp(p[2] + (i == BAR_BIGGER ? 0.1f : -0.1f), 0.5f, 2.5f);
                    }
                    break;
                case BAR_EDIT:
                    if (selected != null) host.editControl(selected);
                    break;
                case BAR_ADD: {
                    int k = cfg.freeCombo();
                    if (k < 0) {
                        hint = "All " + Controls.COMBOS + " added buttons are in use: clear one in Settings → Combos.";
                        break;
                    }
                    Ctl c = Ctl.values()[Ctl.COMBO1.ordinal() + k];
                    float[] p = cfg.place[c.ordinal()];
                    p[0] = 0.5f;
                    p[1] = 0.5f;
                    p[2] = 1;
                    cfg.shown[c.ordinal()] = true;
                    selected = c;
                    host.addButton(k);
                    break;
                }
                case BAR_DELETE:
                    if (selected == null) {
                        hint = "Tap a button you added, then Delete.";
                    } else if (!selected.combo()) {
                        hint = "Only buttons you added can be deleted: Hide takes this one off the screen.";
                    } else {
                        cfg.removeButton(selected.comboIndex());
                        selected = null;
                        host.buttonRemoved();
                    }
                    break;
                case BAR_HIDE:
                    if (selected != null && selected != Ctl.MENU) cfg.shown[selected.ordinal()] = !cfg.shown[selected.ordinal()];
                    break;
                case BAR_GRID:
                    cfg.grid = !cfg.grid;
                    break;
                case BAR_RESET:
                    cfg.resetLayout();
                    break;
                default:
                    editing = false;
                    selected = null;
                    host.editDone();
                    break;
            }
            if (i != BAR_ADD && i != BAR_DELETE) hint = null;
            if (editing) host.layoutChanged();
            host.tick();
            invalidate();
            return true;
        }
        return false;
    }
    private String hint;  // a message under the bar instead of the usual one (until the next tap on the bar)

    // ---- dibujo -------------------------------------------------------------------------------------------------------

    @Override
    protected void onDraw(Canvas c) {
        if (dirtyGlyphs == -1L) syncAppearance(null);
        int alpha = Math.round(255 * cfg.opacity / 100f);
        if (editing && cfg.grid) {  // (faint lines, a stronger one every fifth)
            float g = gridStep();
            int n = 0;
            for (float gx = 0; gx <= getWidth(); gx += g, n++) {
                barPaint.setColor(n % 5 == 0 ? 0x50C49646 : 0x24C49646);
                c.drawRect(gx, 0, gx + 1, getHeight(), barPaint);
            }
            n = 0;
            for (float gy = 0; gy <= getHeight(); gy += g, n++) {
                barPaint.setColor(n % 5 == 0 ? 0x50C49646 : 0x24C49646);
                c.drawRect(0, gy, getWidth(), gy + 1, barPaint);
            }
        }
        for (Ctl ctl : CONTROLS) {
            if (!active(ctl)) continue;
            int a = alpha * cfg.alpha[ctl.ordinal()] / 100;
            if (editing && !cfg.shown[ctl.ordinal()]) a = a / 3;  // (escondido: se ve tenue, para poder enseñarlo)
            if (editing && ctl == selected) a = 255;
            if (editing || !c.isHardwareAccelerated()) draw(c, ctl, a);
            else drawRetained(c, ctl, a);
            if (editing && ctl == selected) {  // (el elegido: un marco dorado de puntos)
                barPaint.setStyle(Paint.Style.STROKE);
                barPaint.setStrokeWidth(getHeight() * 0.004f);
                barPaint.setColor(Glyphs.GOLD);
                barPaint.setPathEffect(new android.graphics.DashPathEffect(new float[] {12, 8}, 0));
                float r = radius(ctl) * 1.25f;
                c.drawCircle(cx(ctl), cy(ctl), r, barPaint);
                barPaint.setPathEffect(null);
            }
        }
        if (editing) drawBar(c);
        if (!editing && c.isHardwareAccelerated()) dirtyGlyphs = 0;
    }

    private void drawRetained(Canvas canvas, Ctl ctl, int alpha) {
        int k = ctl.ordinal();
        Appearance appearance = artwork[k];
        if (appearance.node == null) appearance.node = new RenderNode(ctl.name());
        RenderNode node = appearance.node;
        if ((dirtyGlyphs & (1L << k)) != 0 || !node.hasDisplayList()) {
            node.setPosition(0, 0, getWidth(), getHeight());
            Canvas recording = node.beginRecording();
            try {
                draw(recording, ctl, alpha);
            } finally {
                node.endRecording();
            }
        }
        canvas.drawRenderNode(node);
    }

    private void draw(Canvas c, Ctl ctl, int a) {
        float x = cx(ctl), y = cy(ctl), r = radius(ctl);
        final int k = ctl.ordinal();
        Appearance state = artwork[k];
        boolean down = state.down;
        android.graphics.Bitmap picture = cfg.image[k] ? bitmap(ctl) : null;
        if (ctl == Ctl.MOD) {  // (its own text, or the look it was given)
            glyphs.custom(c, x, y, r, cfg.shape[k] == 0 ? 1 : cfg.shape[k], cfg.icon[k],
                    cfg.label[k] != null ? cfg.label[k] : "MOD", false, down, a, Controls.COLOR_FILLS[cfg.color[k]], picture,
                    cfg.outline[k] / 100f, cfg.fit[k]);
            return;
        }
        if (ctl == Ctl.MENU && cfg.styled(ctl)) {  // (its gear, unless it has an icon or a label)
            glyphs.custom(c, x, y, r, cfg.shape[k] == 0 ? 1 : cfg.shape[k], cfg.icon[k], cfg.label[k], false, down, a,
                    Controls.COLOR_FILLS[cfg.color[k]], picture, cfg.outline[k] / 100f, cfg.fit[k]);
            return;
        }
        if (Controls.isButton(ctl) && (cfg.styled(ctl) || single(ctl))) {
            // (a button with a look of its own; a combo of one button looks like that button; tap to hold shows a dot)
            int shape = cfg.shape[k];
            if (shape == 0) shape = ctl.combo() && !single(ctl) ? 2 : 1;
            String text = cfg.label[k] != null ? cfg.label[k]
                    : ctl.combo() ? Controls.names(cfg.combo[ctl.comboIndex()]) : buttonText(ctl);
            glyphs.custom(c, x, y, r, shape, cfg.icon[k], text, cfg.hold[k], down, a, Controls.COLOR_FILLS[cfg.color[k]],
                    picture, cfg.outline[k] / 100f, cfg.fit[k]);
            return;
        }
        switch (ctl) {
            case LSTICK:
                glyphs.stick(c, x, y, r, state.x, state.y, state.held, state.click, a, cfg.shape[k],
                        Controls.COLOR_FILLS[cfg.color[k]], cfg.outline[k] / 100f, picture, cfg.fit[k]);
                break;
            case RSTICK:
                glyphs.stick(c, x, y, r, state.x, state.y, state.held, state.click, a, cfg.shape[k],
                        Controls.COLOR_FILLS[cfg.color[k]], cfg.outline[k] / 100f, picture, cfg.fit[k]);
                break;
            case DPAD:
                glyphs.dpad(c, x, y, r, state.dirs, a, Controls.COLOR_FILLS[cfg.color[k]], cfg.outline[k] / 100f, picture,
                        cfg.fit[k]);
                break;
            case A: case B: case X: case Y:
                glyphs.face(c, x, y, r, ctl.name(), down, a);
                break;
            case L: case R:
                glyphs.shoulder(c, x, y, r * 2, r * 0.9f, ctl.name(), false, ctl == Ctl.L, down, a);
                break;
            case ZL: case ZR:
                glyphs.shoulder(c, x, y, r * 2, r * 1.3f, ctl.name(), true, ctl == Ctl.ZL, down, a);
                break;
            case MINUS:
                glyphs.small(c, x, y, r, Glyphs.Small.MINUS, down, a);
                break;
            case PLUS:
                glyphs.small(c, x, y, r, Glyphs.Small.PLUS, down, a);
                break;
            case MENU:
                glyphs.small(c, x, y, r, Glyphs.Small.MENU, down, a);
                break;
            case L3:
                glyphs.small(c, x, y, r, Glyphs.Small.L3, down || state.click, a);
                break;
            case R3:
                glyphs.small(c, x, y, r, Glyphs.Small.R3, down || state.click, a);
                break;
            default:
                glyphs.combo(c, x, y, r, Controls.names(cfg.combo[ctl.comboIndex()]), down, a);
                break;
        }
    }

    private boolean single(Ctl ctl) { return ctl.combo() && Long.bitCount(cfg.combo[ctl.comboIndex()]) == 1; }

    private static String buttonText(Ctl c) {
        switch (c) {
            case MINUS: return "−";
            case PLUS: return "+";
            default: return c.name();
        }
    }

    private boolean pressed(Ctl ctl) {
        for (Finger f : fingers) if (f.id >= 0 && f.ctl == ctl && !editing) return ctl != Ctl.MENU || !f.moved;
        return false;
    }
    private int dpadHeld() {
        int d = 0;
        for (Finger f : fingers) if (f.id >= 0 && f.ctl == Ctl.DPAD) d |= f.dirs;
        return d;
    }

    private void drawBar(Canvas c) {
        barText.setTextSize(getHeight() * 0.034f);
        barText.setTypeface(android.graphics.Typeface.create(android.graphics.Typeface.SERIF, android.graphics.Typeface.BOLD));
        for (int i = 0; i < BAR_ITEMS; i++) {
            RectF b = barItem(i);
            barPaint.setStyle(Paint.Style.FILL);
            barPaint.setColor(i == BAR_DONE ? 0xF0E2B65C : 0xF0281010);
            c.drawRect(b, barPaint);
            barPaint.setStyle(Paint.Style.STROKE);
            barPaint.setStrokeWidth(getHeight() * 0.003f);
            barPaint.setColor(Glyphs.BRONZE);
            c.drawRect(b, barPaint);
            barText.setColor(i == BAR_DONE ? Glyphs.DEEP : Glyphs.CREAM);
            Paint.FontMetrics m = barText.getFontMetrics();
            c.drawText(barLabel(i), b.centerX(), b.centerY() - (m.ascent + m.descent) / 2, barText);
        }
        // debajo, qué hacer
        barText.setTextSize(getHeight() * 0.03f);
        barText.setColor(Glyphs.CREAM);
        String hint = this.hint != null ? this.hint
                : selected == null ? "Drag a control to move it; tap it to choose it, then Edit for its size and look."
                                   : "Chosen: " + label(selected);
        rect.set(barItem(0).left, barItem(0).bottom, barItem(BAR_ITEMS - 1).right, barItem(0).bottom + getHeight() * 0.06f);
        c.drawText(hint, rect.centerX(), rect.centerY() + getHeight() * 0.012f, barText);
    }

    private String label(Ctl c) {
        String name = cfg.name(c);
        return c.combo() || name.length() <= 2 ? name : name.toLowerCase(java.util.Locale.ROOT);
    }
}
