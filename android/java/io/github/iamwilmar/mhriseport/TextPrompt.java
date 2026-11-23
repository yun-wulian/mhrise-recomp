package io.github.iamwilmar.mhriseport;

import android.app.Dialog;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.ColorDrawable;
import android.graphics.drawable.GradientDrawable;
import android.text.InputFilter;
import android.text.InputType;
import android.util.DisplayMetrics;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.view.inputmethod.EditorInfo;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;


// The console's software keyboard (nn::swkbd, runtime/src/nnsdk/swkbd.cpp): hunter, Palico and Palamute names, and the
// chat. The Switch opens its keyboard applet; here it's a text box with the Android keyboard, styled like the settings
// menu. Whatever text the game passes in is already filled in. OK (or the keyboard's Done key) sends it back, Cancel or
// Back cancels. If the game rejects the text (a name with words it doesn't allow) the box comes back with its message.
final class TextPrompt {
    interface Answer {
        void done(String text);  // null: cancelled
    }

    private final GameActivity activity;
    private final float scale;
    private final boolean chinese;

    TextPrompt(GameActivity activity) {
        this(activity, false);
    }

    TextPrompt(GameActivity activity, boolean chinese) {
        this.activity = activity;
        this.chinese = chinese;
        this.scale = SettingsMenu.sizeFactor(activity);
    }

    void show(String header, String guide, String initial, String error, int minLength, int maxLength, boolean password,
              boolean multiline, Answer answer) {
        final Dialog d = new Dialog(activity);
        d.requestWindowFeature(Window.FEATURE_NO_TITLE);
        LinearLayout box = new LinearLayout(activity);
        box.setOrientation(LinearLayout.VERTICAL);
        box.setPadding(dp(16), dp(12), dp(16), dp(14));
        box.setBackground(plate(SettingsMenu.PANEL, 2, 0));
        TextView title = text(header.isEmpty() ? (chinese ? "输入" : "Type") : header, 20, SettingsMenu.GOLD, true);
        title.setTypeface(Typeface.create(Typeface.SERIF, Typeface.BOLD));
        box.addView(title);
        box.addView(rule(SettingsMenu.RED, 2));
        box.addView(rule(SettingsMenu.GREEN, 1));

        final EditText edit = new EditText(activity);
        edit.setTextSize(TypedValue.COMPLEX_UNIT_SP, 18 * scale);
        edit.setTextColor(SettingsMenu.CREAM);
        edit.setHintTextColor(SettingsMenu.DIM);
        edit.setBackground(plate(SettingsMenu.ROW, 1, 2));
        edit.setPadding(dp(10), dp(8), dp(10), dp(8));
        int type = InputType.TYPE_CLASS_TEXT;
        if (password) type |= InputType.TYPE_TEXT_VARIATION_PASSWORD;
        if (multiline) type |= InputType.TYPE_TEXT_FLAG_MULTI_LINE;
        edit.setInputType(type);
        edit.setImeOptions(EditorInfo.IME_ACTION_DONE | EditorInfo.IME_FLAG_NO_EXTRACT_UI);
        if (maxLength > 0) edit.setFilters(new InputFilter[] {new InputFilter.LengthFilter(maxLength)});
        if (!guide.isEmpty()) edit.setHint(guide);
        edit.setText(initial);
        edit.setSelection(edit.length());
        LinearLayout.LayoutParams field = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
        field.setMargins(0, dp(10), 0, dp(4));
        box.addView(edit, field);

        final TextView note = text(error, 14, 0xFFE07060, false);
        if (error.isEmpty()) note.setVisibility(TextView.GONE);
        box.addView(note);
        if (maxLength > 0) {
            String limits = minLength > 1 ? minLength + " to " + maxLength + " characters"
                    : "Up to " + maxLength + " characters";
            if (chinese) limits = minLength > 1 ? minLength + " 至 " + maxLength + " 个字符" : "最多 " + maxLength + " 个字符";
            box.addView(text(limits, 13, SettingsMenu.DIM, false));
        }

        final boolean[] answered = {false};
        Runnable accept = () -> {
            String s = edit.getText().toString();
            if (s.length() < Math.max(minLength, 0)) {
                note.setText(chinese ? "至少需要 " + minLength + " 个字符。" : "It needs at least " + minLength + " characters.");
                note.setVisibility(TextView.VISIBLE);
                return;
            }
            answered[0] = true;
            answer.done(s);
            d.dismiss();
        };
        edit.setOnEditorActionListener((v, id, event) -> {
            if (id != EditorInfo.IME_ACTION_DONE) return false;
            accept.run();
            return true;
        });

        LinearLayout buttons = new LinearLayout(activity);
        buttons.setPadding(0, dp(12), 0, 0);
        TextView cancel = button(chinese ? "取消" : "Cancel", false);
        cancel.setOnClickListener(v -> d.cancel());
        TextView ok = button(chinese ? "确定" : "OK", true);
        ok.setOnClickListener(v -> accept.run());
        LinearLayout.LayoutParams half = new LinearLayout.LayoutParams(0, dp(46), 1);
        half.setMargins(dp(4), 0, dp(4), 0);
        buttons.addView(cancel, half);
        buttons.addView(ok, new LinearLayout.LayoutParams(half));
        box.addView(buttons);

        // Cancel, Back or a tap outside: the game gets "cancelled" (once, however the box closes)
        d.setOnDismissListener(x -> {
            if (answered[0]) return;
            answered[0] = true;
            answer.done(null);
        });
        d.setContentView(box);
        Window w = d.getWindow();
        if (w != null) {
            w.setBackgroundDrawable(new ColorDrawable(Color.TRANSPARENT));
            DisplayMetrics m = activity.getResources().getDisplayMetrics();
            w.setLayout(Math.min(dp(620), (int) (m.widthPixels * 0.85f)), ViewGroup.LayoutParams.WRAP_CONTENT);
            w.setGravity(Gravity.TOP | Gravity.CENTER_HORIZONTAL);  // at the top: the Android keyboard takes the bottom half
            w.setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_STATE_ALWAYS_VISIBLE
                    | WindowManager.LayoutParams.SOFT_INPUT_ADJUST_PAN);
        }
        d.show();
        edit.requestFocus();
    }

    // ---- pieces (same as the settings menu) ------------------------------------------------------------------------

    private TextView button(String s, boolean main) {
        TextView b = text(s, 16, main ? 0xFF281010 : SettingsMenu.CREAM, true);
        b.setGravity(Gravity.CENTER);
        b.setBackground(plate(main ? SettingsMenu.GOLD : 0xFF3E2616, 1, 4));
        b.setFocusable(true);  // (so the d-pad can reach the buttons on a controller)
        return b;
    }

    private android.view.View rule(int color, int heightDp) {
        android.view.View v = new android.view.View(activity);
        v.setBackgroundColor(color);
        v.setLayoutParams(new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(heightDp)));
        return v;
    }

    private GradientDrawable plate(int fill, int strokeDp, int radiusDp) {
        GradientDrawable g = new GradientDrawable();
        g.setColor(fill);
        g.setStroke(dp(strokeDp), SettingsMenu.EDGE);
        g.setCornerRadius(dp(radiusDp));
        return g;
    }

    private TextView text(String s, int sp, int color, boolean bold) {
        TextView t = new TextView(activity);
        t.setText(s);
        t.setTextSize(TypedValue.COMPLEX_UNIT_SP, sp * scale);
        t.setTextColor(color);
        if (bold) t.setTypeface(Typeface.DEFAULT_BOLD);
        return t;
    }

    private int dp(int v) {
        float px = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v * scale, activity.getResources().getDisplayMetrics());
        return v > 0 ? Math.max(1, Math.round(px)) : Math.round(px);
    }
}
