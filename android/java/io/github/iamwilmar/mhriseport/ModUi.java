package io.github.iamwilmar.mhriseport;

import android.graphics.PixelFormat;
import android.opengl.GLSurfaceView;
import android.view.MotionEvent;
import android.view.KeyEvent;
import android.view.SurfaceHolder;
import android.view.WindowManager;
import android.view.inputmethod.BaseInputConnection;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputConnection;
import android.view.inputmethod.InputMethodManager;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

final class ModUi extends GLSurfaceView implements GLSurfaceView.Renderer {
    private static native void nativeCreate(float scale);
    private static native void nativeScale(float scale);
    private static native void nativeOpacity(float alpha);
    private static native int nativeFrame(int width,int height);
    private static native void nativeSize(int width,int height);
    private static native boolean nativePending();
    private static native void nativeMenu(boolean open);
    private static native void nativeTouch(float x,float y,boolean down);
    private static native boolean nativeWindowAt(float x, float y);
    private static native void nativeChar(int codepoint);
    private static native void nativeClose();
    private int width,height;
    private volatile boolean menu, keyboard;
    private boolean created;
    private final int[] screenOrigin = new int[2];
    private int pointer = MotionEvent.INVALID_POINTER_ID;
    private final GameActivity activity;
    private final WindowManager.LayoutParams panel;
    private final Runnable poll = new Runnable() {
        public void run() {
            if (menu && !nativeMenuOpen()) setMenu(false);
            if (nativePending()) requestRender();
            postOnAnimation(this);
        }
    };
    ModUi(GameActivity a) {
        super(a);
        activity = a;
        setEGLContextClientVersion(3);
        setEGLConfigChooser(8, 8, 8, 8, 0, 0);
        getHolder().setFormat(PixelFormat.TRANSLUCENT);
        setZOrderOnTop(true);
        setFocusableInTouchMode(true);
        setRenderer(this);
        setRenderMode(RENDERMODE_WHEN_DIRTY);
        panel = new WindowManager.LayoutParams(-1, -1, WindowManager.LayoutParams.TYPE_APPLICATION_PANEL,
                WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN | WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS
                | WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE | WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE,
                PixelFormat.TRANSLUCENT);
        panel.layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_ALWAYS;
        panel.token = a.getWindow().getDecorView().getWindowToken();
        panel.setTitle("Lua HUD");
        a.getWindowManager().addView(this, panel);
    }
    boolean menuOpen() {
        return menu;
    }

    // The controller hotkey (mod_ui_hotkey in mhr.ini: Android key codes joined by "+"). Unset: L3 + R3, as in
    // REFramework. "0": none. android_main.cpp reads the same line to open the menu; while it's open, this window has
    // the keys, so it closes it here.
    static final String HOTKEY = "mod_ui_hotkey";
    static final String OPACITY = "mod_ui_opacity";  // (percent: how much of the hunt the window hides)
    static int opacity(Ini ini) {
        try {
            return Math.max(30, Math.min(100, Integer.parseInt(ini.get(OPACITY).trim())));
        } catch (RuntimeException e) {
            return 94;
        }
    }
    static final int[] DEFAULT_HOTKEY = {KeyEvent.KEYCODE_BUTTON_THUMBL, KeyEvent.KEYCODE_BUTTON_THUMBR};

    static int[] hotkey(Ini ini) {
        String v = ini.get(HOTKEY);
        if (v == null) return DEFAULT_HOTKEY;
        java.util.List<Integer> keys = new java.util.ArrayList<>();
        for (String part : v.split("\\+")) {
            try {
                int k = Integer.parseInt(part.trim());
                if (k > 0) keys.add(k);
            } catch (NumberFormatException ignored) {
            }
        }
        int[] out = new int[keys.size()];
        for (int i = 0; i < out.length; i++) out[i] = keys.get(i);
        return out;
    }

    private final java.util.Set<Integer> held = new java.util.HashSet<>();
    private int[] menuHotkey = DEFAULT_HOTKEY;

    void setMenu(boolean open) {
        if (menu == open) return;
        cancelPointer();
        menu = open;
        held.clear();
        if (open) {
            java.io.File folder = SettingsMenu.gameFolder(activity);
            if (folder != null) {
                Ini ini = new Ini(folder);
                menuHotkey = hotkey(ini);
                nativeOpacity(opacity(ini) / 100f);
            }
        }
        if (open) nativeScale(uiScale());  // (Menu size may have changed since)
        nativeMenu(open);
        int pass = WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE | WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE;
        if (open) panel.flags &= ~pass;
        else panel.flags |= pass;
        activity.getWindowManager().updateViewLayout(this, panel);
        if (open) requestFocus();
        else setKeyboard(false);
    }
    void dispose() {
        nativeMenu(false);
        activity.getWindowManager().removeViewImmediate(this);
    }
    @Override public boolean onKeyDown(int key, KeyEvent event) {
        if (menu && event.getRepeatCount() == 0 && menuHotkey.length > 0) {
            held.add(key);
            boolean all = true;
            for (int k : menuHotkey) all &= held.contains(k);
            if (all) {
                setMenu(false);
                return true;
            }
        }
        return super.onKeyDown(key, event);
    }
    @Override public boolean onKeyUp(int key, KeyEvent event) {
        held.remove(key);
        if (menu && key == KeyEvent.KEYCODE_BACK) {
            setMenu(false);
            return true;
        }
        return super.onKeyUp(key, event);
    }
    @Override public void onResume() {
        super.onResume();
        removeCallbacks(poll);
        postOnAnimation(poll);
    }
    @Override public void onPause() {
        cancelPointer();
        removeCallbacks(poll);
        queueEvent(this::closeRenderer);
        super.onPause();
    }
    @Override protected void onDetachedFromWindow() {
        removeCallbacks(poll);
        super.onDetachedFromWindow();
    }
    private void closeRenderer() {
        if (created) nativeClose();
        created = false;
    }
    // The settings menu's size: pixels per dp, the screen's share and Menu size
    private float uiScale() {
        return getResources().getDisplayMetrics().density * SettingsMenu.sizeFactor(getContext());
    }
    public void onSurfaceCreated(GL10 gl,EGLConfig config) {
        closeRenderer();
        nativeCreate(uiScale());
        created = true;
    }
    @Override public void surfaceDestroyed(SurfaceHolder holder){
        queueEvent(this::closeRenderer);
        super.surfaceDestroyed(holder);
    }
    public void onSurfaceChanged(GL10 gl,int w,int h){
        if (!created) onSurfaceCreated(gl,null);
        width=w;height=h;nativeSize(w,h);
    }
    public void onDrawFrame(GL10 gl){
        int state=nativeFrame(width,height);
        boolean wants = (state & 2) != 0;
        if (menu && wants != keyboard)
            post(() -> { if (isAttachedToWindow() && menu) setKeyboard(wants); });
    }
    private static native boolean nativeMenuOpen();
    private static native int nativeRuntime();
    // 0: no Lua runtime (no scripts), 1: scripts loaded, not running yet, 2: running
    static int runtime() {
        try {
            return nativeRuntime();
        } catch (UnsatisfiedLinkError e) {
            return 0;
        }
    }
    private void setKeyboard(boolean wants) {
        if (keyboard == wants) return;
        keyboard = wants;
        InputMethodManager im=getContext().getSystemService(InputMethodManager.class);
        if(wants)im.showSoftInput(this,InputMethodManager.SHOW_IMPLICIT);
        else im.hideSoftInputFromWindow(getWindowToken(),0);
    }
    private void cancelPointer() {
        if (pointer != MotionEvent.INVALID_POINTER_ID) {
            nativeTouch(-Float.MAX_VALUE, -Float.MAX_VALUE, false);
            pointer = MotionEvent.INVALID_POINTER_ID;
        }
    }
    boolean routeTouch(int action, int id, float screenX, float screenY) {
        getLocationOnScreen(screenOrigin);
        float x = screenX - screenOrigin[0], y = screenY - screenOrigin[1];
        if (action == MotionEvent.ACTION_DOWN) {
            if (!menu && !nativeWindowAt(x, y)) return false;
            // ImGui has one mouse. Extra fingers on a window stay off the gamepad.
            if (pointer != MotionEvent.INVALID_POINTER_ID) return true;
            pointer = id;
        }
        if (pointer == id) {
            if (action == MotionEvent.ACTION_CANCEL) cancelPointer();
            else {
                boolean down = action != MotionEvent.ACTION_UP;
                nativeTouch(x, y, down);
                if (!down) pointer = MotionEvent.INVALID_POINTER_ID;
            }
        }
        return true;
    }
    @Override public boolean onTouchEvent(MotionEvent ev) {
        int action = ev.getActionMasked();
        if (action == MotionEvent.ACTION_CANCEL) {
            cancelPointer();
            return true;
        }
        int index = action == MotionEvent.ACTION_MOVE ? ev.findPointerIndex(pointer) : ev.getActionIndex();
        if (action == MotionEvent.ACTION_POINTER_DOWN) action = MotionEvent.ACTION_DOWN;
        if (action == MotionEvent.ACTION_POINTER_UP) action = MotionEvent.ACTION_UP;
        if (index >= 0) routeTouch(action, ev.getPointerId(index), ev.getRawX(index), ev.getRawY(index));
        return true;
    }
    @Override public boolean onCheckIsTextEditor(){return true;}
    @Override public InputConnection onCreateInputConnection(EditorInfo info){
        info.inputType=android.text.InputType.TYPE_CLASS_TEXT;
        return new BaseInputConnection(this,false){
            @Override public boolean commitText(CharSequence s,int pos){
                int[] chars=s.toString().codePoints().toArray();queueEvent(()->{for(int c:chars)nativeChar(c);});return true;
            }
            @Override public boolean deleteSurroundingText(int before,int after){
                queueEvent(()->{for(int i=0;i<before;i++)nativeChar(8);});return true;
            }
        };
    }
}
