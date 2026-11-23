#include "loader/mod_ui.h"
#include "loader/mod_font.h"
#include "loader/lua_values.h"
#include "loader/screen_projection.h"
#include "loader/application_lua.h"
#include <algorithm>
#include <GLES3/gl3.h>
#include <android/log.h>
#include <atomic>
#include <cfloat>
#include <backends/imgui_impl_opengl3.h>
#include <chrono>
#include <cstdio>
#include <imgui.h>
#include <imgui_internal.h>
#include <jni.h>
#include <lua.hpp>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {
thread_local bool drawing = false;
enum class Scope { Tree, Tooltip, Window, Id, Style, Group, Rect, Disabled, Popup, ItemWidth,
                   Child, ListBox, MenuBar, MainMenuBar, Menu, Table, StyleColor };
thread_local std::vector<Scope> scopes;
void close_scope() {
  switch (scopes.back()) {
  case Scope::Tree: ImGui::TreePop(); break;
  case Scope::Tooltip: ImGui::EndTooltip(); break;
  case Scope::Window: ImGui::End(); break;
  case Scope::Id: ImGui::PopID(); break;
  case Scope::Style: ImGui::PopStyleVar(); break;
  case Scope::Group:
  case Scope::Rect: ImGui::EndGroup(); break;
  case Scope::Disabled: ImGui::EndDisabled(); break;
  case Scope::Popup: ImGui::EndPopup(); break;
  case Scope::ItemWidth: ImGui::PopItemWidth(); break;
  case Scope::Child: ImGui::EndChild(); break;
  case Scope::ListBox: ImGui::EndListBox(); break;
  case Scope::MenuBar: ImGui::EndMenuBar(); break;
  case Scope::MainMenuBar: ImGui::EndMainMenuBar(); break;
  case Scope::Menu: ImGui::EndMenu(); break;
  case Scope::Table: ImGui::EndTable(); break;
  case Scope::StyleColor: ImGui::PopStyleColor(); break;
  }
  scopes.pop_back();
}
ImGuiContext *ui = nullptr;
std::atomic<bool> visible = false;
std::atomic<bool> needs_render = false;
bool had_geometry = false;
const char *visible_text(const char *text) {
  if (auto *glyphs = static_cast<ModFontGlyphs *>(ImGui::GetIO().UserData))
    glyphs->observe(text);
  return text;
}
enum Widget {
  Text,
  TextColored,
  SameLine,
  Spacing,
  Separator,
  Tree,
  Pop,
  Button,
  Check,
  DragInt,
  SliderInt,
  DragFloat,
  Slider,
  Input,
  NewLine,
  Hovered,
  Tooltip,
  EndTooltip,
  BeginWindow,
  EndWindow,
  NextPos,
  NextSize,
  DisplaySize,
  WindowPos,
  MouseDown,
  PushId,
  PopId,
  PushStyle,
  PopStyle,
  Indent,
  Unindent,
  Combo, CalcTextSize, WindowSize, CursorScreenPos, CursorPos, SetCursorPos,
  SetCursorScreenPos, ItemActive, ItemFocused, SetNextItemWidth, CalcItemWidth,
  PushItemWidth, PopItemWidth, BeginGroup, EndGroup, BeginRect, EndRect,
  BeginDisabled, EndDisabled, OpenPopup, BeginPopup, BeginPopupContextItem,
  EndPopup, CloseCurrentPopup, IsPopupOpen, InputMultiline, ColorPickerArgb,
  // The rest of REFramework's imgui table
  BeginChild, EndChild, SmallButton, InvisibleButton, ArrowButton, DragFloat2, DragFloat3, DragFloat4,
  ColorPicker, ColorPicker3, ColorPicker4, ColorEdit, ColorEditArgb, ColorEdit3, ColorEdit4,
  TreeNodePtr, TreeNodeStr, CollapsingHeader, LoadFont, PushFont, PopFont, DefaultFontSize, GetMouse,
  ProgressBar, KeyIndex, KeyDown, KeyPressed, KeyReleased, MouseClicked, MouseReleased, MouseDoubleClicked,
  SetTooltip, NextItemOpen, BeginListBox, EndListBox, BeginMenuBar, EndMenuBar, BeginMainMenuBar,
  EndMainMenuBar, BeginMenu, EndMenu, MenuItem, PushStyleColor, PopStyleColor, CursorStartPos,
  SetItemDefaultFocus, ScrollX, ScrollY, SetScrollX, SetScrollY, ScrollMaxX, ScrollMaxY, SetScrollHereX,
  SetScrollHereY, SetScrollFromPosX, SetScrollFromPosY, BeginTable, EndTable, TableNextRow, TableNextColumn,
  TableSetColumnIndex, TableSetupColumn, TableSetupScrollFreeze, TableHeadersRow, TableHeader,
  TableColumnCount, TableColumnIndex, TableRowIndex, TableColumnName, TableColumnFlags, TableSetBgColor,
  SetClipboard, GetClipboard, PushFontSize, PopFontSize
};
ImVec2 vec2(lua_State *L, int index) {
  float v[2];
  if (mod_lua::read_components(L, index, false, v, 2))
    return {v[0], v[1]};
  luaL_checktype(L, index, LUA_TTABLE);
  for (int axis = 0; axis < 2; ++axis) {
    lua_rawgeti(L, index, axis + 1);
    if (lua_isnil(L, -1)) {
      lua_pop(L, 1);
      lua_getfield(L, index, axis ? "y" : "x");
    }
    v[axis] = luaL_checknumber(L, -1);
    lua_pop(L, 1);
  }
  return {v[0], v[1]};
}
// A Vector2f/3f/4f, or a table with x, y, z, w (or 1..4)
void vecn(lua_State *L, int index, float *v, int n) {
  if (mod_lua::read_components(L, index, false, v, n))
    return;
  luaL_checktype(L, index, LUA_TTABLE);
  static const char *const axes[] = {"x", "y", "z", "w"};
  for (int i = 0; i < n; ++i) {
    lua_rawgeti(L, index, i + 1);
    if (lua_isnil(L, -1)) {
      lua_pop(L, 1);
      lua_getfield(L, index, axes[i]);
    }
    v[i] = luaL_checknumber(L, -1);
    lua_pop(L, 1);
  }
}
// ImU32 colors are ABGR (as REFramework's color_picker / color_edit take them); the _argb ones are ARGB
ImVec4 from_argb(uint32_t argb) {
  return {float((argb >> 16) & 255) / 255, float((argb >> 8) & 255) / 255, float(argb & 255) / 255,
          float(argb >> 24) / 255};
}
uint32_t to_argb(const ImVec4 &c) {
  auto channel = [](float v) { return uint32_t(std::clamp(v, 0.f, 1.f) * 255.f + .5f); };
  return channel(c.w) << 24 | channel(c.x) << 16 | channel(c.y) << 8 | channel(c.z);
}
// Scripts check a key by ImGuiKey; anything else isn't a key here (there's no keyboard on most phones anyway)
bool key_state(lua_State *L, int which) {
  auto key = static_cast<ImGuiKey>(luaL_checkinteger(L, 1));
  if (key < ImGuiKey_NamedKey_BEGIN || key >= ImGuiKey_NamedKey_END)
    return false;
  return which == 0 ? ImGui::IsKeyDown(key) : which == 1 ? ImGui::IsKeyPressed(key) : ImGui::IsKeyReleased(key);
}
int more_widgets(lua_State *L, Widget kind);
int widget(lua_State *L) {
  if (!drawing)
    return luaL_error(L, "imgui requires an active on_frame or on_draw_ui callback");
  auto kind = static_cast<Widget>(lua_tointeger(L, lua_upvalueindex(1)));
  auto end = [&](Scope scope) {
    if (scopes.empty() || scopes.back() != scope)
      return luaL_error(L, "ImGui scope closed out of order");
    close_scope();
    return 0;
  };
  switch (kind) {
  case BeginGroup:
  case BeginRect:
    ImGui::BeginGroup();
    scopes.push_back(kind == BeginRect ? Scope::Rect : Scope::Group);
    return 0;
  case EndGroup: return end(Scope::Group);
  case EndRect: {
    float pad = luaL_optnumber(L, 1, 0);
    float radius = luaL_optnumber(L, 2, ImGui::GetStyle().FrameRounding);
    end(Scope::Rect);
    auto lo = ImGui::GetItemRectMin(), hi = ImGui::GetItemRectMax();
    ImGui::GetWindowDrawList()->AddRect({lo.x-pad, lo.y-pad}, {hi.x+pad, hi.y+pad},
                                       ImGui::GetColorU32(ImGuiCol_Border), radius);
    return 0;
  }
  case BeginDisabled:
    ImGui::BeginDisabled(lua_isnoneornil(L, 1) || lua_toboolean(L, 1));
    scopes.push_back(Scope::Disabled);
    return 0;
  case EndDisabled: return end(Scope::Disabled);
  case BeginPopupContextItem:
  case BeginPopup: {
    bool open = kind == BeginPopup
        ? ImGui::BeginPopup(visible_text(luaL_checkstring(L, 1)), luaL_optinteger(L, 2, 0))
        : ImGui::BeginPopupContextItem(luaL_optstring(L, 1, nullptr),
                                      luaL_optinteger(L, 2, ImGuiPopupFlags_MouseButtonRight));
    if (open) scopes.push_back(Scope::Popup);
    lua_pushboolean(L, open);
    return 1;
  }
  case EndPopup: return end(Scope::Popup);
  case OpenPopup:
    ImGui::OpenPopup(luaL_checkstring(L, 1), luaL_optinteger(L, 2, 0));
    return 0;
  case IsPopupOpen:
    lua_pushboolean(L, ImGui::IsPopupOpen(luaL_checkstring(L, 1)));
    return 1;
  case CloseCurrentPopup:
    ImGui::CloseCurrentPopup();
    return 0;
  case SetCursorScreenPos: ImGui::SetCursorScreenPos(vec2(L, 1)); return 0;
  case SetCursorPos: ImGui::SetCursorPos(vec2(L, 1)); return 0;
  case SetNextItemWidth: ImGui::SetNextItemWidth(luaL_checknumber(L, 1)); return 0;
  case CalcItemWidth: lua_pushnumber(L, ImGui::CalcItemWidth()); return 1;
  case PushItemWidth:
    ImGui::PushItemWidth(luaL_checknumber(L, 1));
    scopes.push_back(Scope::ItemWidth);
    return 0;
  case PopItemWidth: return end(Scope::ItemWidth);
  case ItemActive: lua_pushboolean(L, ImGui::IsItemActive()); return 1;
  case ItemFocused: lua_pushboolean(L, ImGui::IsItemFocused()); return 1;
  case Indent:
    ImGui::Indent(luaL_optnumber(L, 1, 0));
    return 0;
  case Unindent:
    ImGui::Unindent(luaL_optnumber(L, 1, 0));
    return 0;
  case PushId:
    if (lua_isinteger(L, 1)) ImGui::PushID(static_cast<int>(lua_tointeger(L, 1)));
    else ImGui::PushID(luaL_checkstring(L, 1));
    scopes.push_back(Scope::Id);
    return 0;
  case PopId:
    if (scopes.empty() || scopes.back() != Scope::Id)
      return luaL_error(L, "pop_id without matching push_id");
    close_scope();
    return 0;
  case PushStyle: {
    int var = luaL_checkinteger(L, 1);
    if (var < 0 || var >= ImGuiStyleVar_COUNT)
      return luaL_argerror(L, 1, "unknown style variable");
    if (ImGui::GetStyleVarInfo(var)->Count == 2) ImGui::PushStyleVar(var, vec2(L, 2));
    else ImGui::PushStyleVar(var, float(luaL_checknumber(L, 2)));
    scopes.push_back(Scope::Style);
    return 0;
  }
  case PopStyle: {
    int count = luaL_optinteger(L, 1, 1);
    if (count < 0 || count > std::count(scopes.begin(), scopes.end(), Scope::Style))
      return luaL_error(L, "pop_style_var exceeds the number pushed");
    ImGui::PopStyleVar(count);
    while (count--) {
      auto it = std::find(scopes.rbegin(), scopes.rend(), Scope::Style);
      scopes.erase(std::next(it).base());
    }
    return 0;
  }
  case MouseDown: {
    int button = luaL_checkinteger(L, 1);
    if (button < 0 || button >= ImGuiMouseButton_COUNT)
      return luaL_argerror(L, 1, "invalid mouse button");
    lua_pushboolean(L, ImGui::IsMouseDown(button));
    return 1;
  }
  case CursorPos:
  case CursorScreenPos:
  case CalcTextSize:
  case WindowSize:
  case WindowPos:
  case DisplaySize: {
    ImVec2 v;
    switch (kind) {
    case CalcTextSize: v = ImGui::CalcTextSize(visible_text(luaL_checkstring(L, 1))); break;
    case CursorPos: v = ImGui::GetCursorPos(); break;
    case CursorScreenPos: v = ImGui::GetCursorScreenPos(); break;
    case WindowSize: v = ImGui::GetWindowSize(); break;
    case WindowPos: v = ImGui::GetWindowPos(); break;
    default: v = ImGui::GetIO().DisplaySize; break;
    }
    float xy[] = {v.x, v.y};
    mod_lua::push_components(L, false, xy, 2);
    return 1;
  }
  case NextPos:
  case NextSize: {
    auto value = vec2(L, 1);
    int cond = luaL_optinteger(L, 2, 0);
    if (cond != 0 && cond != ImGuiCond_Always && cond != ImGuiCond_Once &&
        cond != ImGuiCond_FirstUseEver && cond != ImGuiCond_Appearing)
      return luaL_argerror(L, 2, "invalid window condition");
    if (kind == NextPos)
      ImGui::SetNextWindowPos(value, cond, lua_isnoneornil(L, 3) ? ImVec2{} : vec2(L, 3));
    else
      ImGui::SetNextWindowSize(value, cond);
    return 0;
  }
  case NewLine:
    ImGui::NewLine();
    return 0;
  case Hovered:
    lua_pushboolean(L, ImGui::IsItemHovered(luaL_optinteger(L, 1, 0)));
    return 1;
  case Tooltip:
    ImGui::BeginTooltip();
    scopes.push_back(Scope::Tooltip);
    return 0;
  case EndTooltip:
    if (scopes.empty() || scopes.back() != Scope::Tooltip)
      return luaL_error(L, "end_tooltip without begin_tooltip");
    close_scope();
    return 0;
  case EndWindow:
    if (scopes.empty() || scopes.back() != Scope::Window)
      return luaL_error(L, "end_window without matching begin_window");
    close_scope();
    return 0;
  case SameLine:
    ImGui::SameLine();
    return 0;
  case Spacing:
    ImGui::Spacing();
    return 0;
  case Separator:
    ImGui::Separator();
    return 0;
  case Pop:
    if (scopes.empty() || scopes.back() != Scope::Tree)
      return luaL_error(L, "tree_pop without an open node");
    close_scope();
    return 0;
  default:
    if (kind >= BeginChild)
      return more_widgets(L, kind);
    break;
  }
  const char *label = visible_text(luaL_checkstring(L, 1));
  switch (kind) {
  case BeginWindow: {
    if (!*label)
      return luaL_error(L, "begin_window requires a nonempty name");
    bool closable = lua_isboolean(L, 2);
    bool open = !closable || lua_toboolean(L, 2);
    if (open) {
      ImGui::Begin(label, closable ? &open : nullptr, luaL_optinteger(L, 3, 0));
      // REFramework returns the open state, including for collapsed windows.
      scopes.push_back(Scope::Window);
    }
    lua_pushboolean(L, open);
    return 1;
  }
  case Combo: {
    luaL_checktype(L, 3, LUA_TTABLE);
    lua_settop(L, 3);
    lua_pushvalue(L, 2);
    lua_rawget(L, 3);
    if (lua_isnil(L, -1)) {
      lua_pop(L, 1);
      lua_pushnil(L);
      if (lua_next(L, 3)) {
        lua_pushvalue(L, -2);
        lua_replace(L, 2);
        lua_remove(L, -2);
      } else
        lua_pushnil(L);
    }
    auto preview = lua_tostring(L, -1);
    bool changed = false;
    if (ImGui::BeginCombo(label, visible_text(preview ? preview : ""))) {
      lua_pushnil(L);
      while (lua_next(L, 3)) {
        if (lua_type(L, -1) == LUA_TSTRING &&
            ImGui::Selectable(visible_text(lua_tostring(L, -1)), lua_rawequal(L, -2, 2))) {
          lua_pushvalue(L, -2);
          lua_replace(L, 2);
          changed = true;
        }
        lua_pop(L, 1);
      }
      ImGui::EndCombo();
    }
    lua_pushboolean(L, changed);
    lua_pushvalue(L, 2);
    return 2;
  }
  case Text:
    ImGui::TextUnformatted(label);
    return 0;
  case TextColored:
    ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(luaL_checkinteger(L, 2)),
                       "%s", label);
    return 0;
  case Button:
    lua_pushboolean(L, ImGui::Button(label, lua_isnoneornil(L, 2) ? ImVec2{} : vec2(L, 2)));
    return 1;
  case Tree: {
    bool open = ImGui::TreeNode(label);
    if (open)
      scopes.push_back(Scope::Tree);
    lua_pushboolean(L, open);
    return 1;
  }
  case Check: {
    bool v = lua_toboolean(L, 2);
    bool changed = ImGui::Checkbox(label, &v);
    lua_pushboolean(L, changed);
    lua_pushboolean(L, v);
    return 2;
  }
  case DragInt: {
    int v = luaL_checkinteger(L, 2);
    bool changed =
        ImGui::DragInt(label, &v, luaL_optnumber(L, 3, 1),
                       luaL_optinteger(L, 4, 0), luaL_optinteger(L, 5, 0), luaL_optstring(L, 6, "%d"));
    lua_pushboolean(L, changed);
    lua_pushinteger(L, v);
    return 2;
  }
  case SliderInt: {
    int v = luaL_checkinteger(L, 2);
    bool changed = ImGui::SliderInt(label, &v, luaL_checkinteger(L, 3),
                                    luaL_checkinteger(L, 4), luaL_optstring(L, 5, "%d"));
    lua_pushboolean(L, changed);
    lua_pushinteger(L, v);
    return 2;
  }
  case DragFloat:
  case Slider: {
    float v = luaL_checknumber(L, 2);
    bool changed = kind == Slider
                       ? ImGui::SliderFloat(label, &v, luaL_checknumber(L, 3),
                                            luaL_checknumber(L, 4), luaL_optstring(L, 5, "%.3f"))
                       : ImGui::DragFloat(label, &v, luaL_optnumber(L, 3, 1),
                                          luaL_optnumber(L, 4, 0),
                                          luaL_optnumber(L, 5, 0), luaL_optstring(L, 6, "%.3f"));
    lua_pushboolean(L, changed);
    lua_pushnumber(L, v);
    return 2;
  }
  case ColorPickerArgb: {
    uint32_t argb = luaL_checkinteger(L, 2);
    float rgba[] = {float((argb >> 16) & 255) / 255, float((argb >> 8) & 255) / 255,
                    float(argb & 255) / 255, float(argb >> 24) / 255};
    bool changed = ImGui::ColorPicker4(label, rgba, luaL_optinteger(L, 3, 0));
    auto channel = [&](int i) { return uint32_t(std::clamp(rgba[i], 0.f, 1.f) * 255.f); };
    lua_pushboolean(L, changed);
    lua_pushinteger(L, (channel(3)<<24) | (channel(0)<<16) | (channel(1)<<8) | channel(2));
    return 2;
  }
  case InputMultiline:
  case Input: {
    size_t n;
    const char *value = visible_text(luaL_optlstring(L, 2, "", &n));
    const bool multi = kind == InputMultiline;
    ImVec2 size = multi && !lua_isnoneornil(L, 3) ? vec2(L, 3) : ImVec2{};
    int flags = luaL_optinteger(L, multi ? 4 : 3, 0) | ImGuiInputTextFlags_CallbackResize |
                ImGuiInputTextFlags_CallbackAlways;
    struct Edit {
      std::vector<char> bytes;
      int from{}, to{};
    } edit;
    edit.bytes.resize(n + 1);
    memcpy(edit.bytes.data(), value, n);
    auto callback = [](ImGuiInputTextCallbackData *data) {
      auto &edit = *static_cast<Edit *>(data->UserData);
      if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
        edit.bytes.resize(data->BufTextLen + 1);
        data->Buf = edit.bytes.data();
      }
      edit.from = data->SelectionStart;
      edit.to = data->SelectionEnd;
      return 0;
    };
    bool changed = multi
        ? ImGui::InputTextMultiline(label, edit.bytes.data(), edit.bytes.size(), size, flags, callback, &edit)
        : ImGui::InputText(label, edit.bytes.data(), edit.bytes.size(), flags, callback, &edit);
    lua_pushboolean(L, changed);
    lua_pushstring(L, edit.bytes.data());
    lua_pushinteger(L, edit.from);
    lua_pushinteger(L, edit.to);
    return 4;
  }
  default:
    return luaL_error(L, "Invalid ImGui widget");
  }
}
int more_widgets(lua_State *L, Widget kind) {
  // The end of a begin that returned false: scripts written for REFramework often call it anyway
  auto end = [&](Scope scope) {
    if (!scopes.empty() && scopes.back() == scope)
      close_scope();
    return 0;
  };
  auto push_bool = [&](bool v) {
    lua_pushboolean(L, v);
    return 1;
  };
  auto push_number = [&](lua_Number v) {
    lua_pushnumber(L, v);
    return 1;
  };
  auto push_vec2 = [&](ImVec2 v) {
    float xy[] = {v.x, v.y};
    mod_lua::push_components(L, false, xy, 2);
    return 1;
  };
  switch (kind) {
  case BeginChild: {
    // REFramework's has no name: one per place in the window
    auto here = ImGui::GetCursorPos();
    auto id = ImGui::GetID(static_cast<int>(here.y) * 4096 + static_cast<int>(here.x));
    bool shown = ImGui::BeginChild(id, lua_isnoneornil(L, 1) ? ImVec2{} : vec2(L, 1),
                                   lua_toboolean(L, 2) ? ImGuiChildFlags(1) : ImGuiChildFlags(0),
                                   luaL_optinteger(L, 3, 0));
    scopes.push_back(Scope::Child);  // (EndChild goes with every BeginChild)
    return push_bool(shown);
  }
  case EndChild: return end(Scope::Child);
  case SmallButton: return push_bool(ImGui::SmallButton(visible_text(luaL_checkstring(L, 1))));
  case InvisibleButton:
    return push_bool(ImGui::InvisibleButton(luaL_checkstring(L, 1), vec2(L, 2), luaL_optinteger(L, 3, 0)));
  case ArrowButton:
    return push_bool(ImGui::ArrowButton(luaL_checkstring(L, 1), static_cast<ImGuiDir>(luaL_checkinteger(L, 2))));
  case DragFloat2:
  case DragFloat3:
  case DragFloat4: {
    int n = kind == DragFloat2 ? 2 : kind == DragFloat3 ? 3 : 4;
    float v[4]{};
    vecn(L, 2, v, n);
    const char *label = visible_text(luaL_checkstring(L, 1));
    float speed = luaL_optnumber(L, 3, 1), lo = luaL_optnumber(L, 4, 0), hi = luaL_optnumber(L, 5, 0);
    const char *format = luaL_optstring(L, 6, "%.3f");
    bool changed = n == 2 ? ImGui::DragFloat2(label, v, speed, lo, hi, format)
                 : n == 3 ? ImGui::DragFloat3(label, v, speed, lo, hi, format)
                          : ImGui::DragFloat4(label, v, speed, lo, hi, format);
    lua_pushboolean(L, changed);
    mod_lua::push_components(L, false, v, n);
    return 2;
  }
  case ColorPicker:
  case ColorEdit:
  case ColorEditArgb: {
    const char *label = visible_text(luaL_checkstring(L, 1));
    auto color = static_cast<uint32_t>(luaL_checkinteger(L, 2));
    ImVec4 c = kind == ColorEditArgb ? from_argb(color) : ImGui::ColorConvertU32ToFloat4(color);
    int flags = luaL_optinteger(L, 3, 0);
    bool changed = kind == ColorPicker ? ImGui::ColorPicker4(label, &c.x, flags) : ImGui::ColorEdit4(label, &c.x, flags);
    lua_pushboolean(L, changed);
    lua_pushinteger(L, kind == ColorEditArgb ? to_argb(c) : ImGui::ColorConvertFloat4ToU32(c));
    return 2;
  }
  case ColorPicker3:
  case ColorPicker4:
  case ColorEdit3:
  case ColorEdit4: {
    const char *label = visible_text(luaL_checkstring(L, 1));
    int n = kind == ColorPicker3 || kind == ColorEdit3 ? 3 : 4;
    float v[4]{0, 0, 0, 1};
    vecn(L, 2, v, n);
    int flags = luaL_optinteger(L, 3, 0);
    bool changed = kind == ColorPicker3 ? ImGui::ColorPicker3(label, v, flags)
                 : kind == ColorPicker4 ? ImGui::ColorPicker4(label, v, flags)
                 : kind == ColorEdit3   ? ImGui::ColorEdit3(label, v, flags)
                                        : ImGui::ColorEdit4(label, v, flags);
    lua_pushboolean(L, changed);
    mod_lua::push_components(L, false, v, n);
    return 2;
  }
  case TreeNodePtr:
  case TreeNodeStr: {
    const char *label = visible_text(luaL_checkstring(L, 2));
    bool open;
    if (kind == TreeNodeStr)
      open = ImGui::TreeNode(luaL_checkstring(L, 1), "%s", label);
    else {
      // Any value as the id: an object's address, a number, a table
      const void *id = lua_isinteger(L, 1) ? reinterpret_cast<const void *>(static_cast<uintptr_t>(lua_tointeger(L, 1)))
                                           : lua_topointer(L, 1);
      open = ImGui::TreeNode(id, "%s", label);
    }
    if (open) scopes.push_back(Scope::Tree);
    return push_bool(open);
  }
  case CollapsingHeader:
    return push_bool(ImGui::CollapsingHeader(visible_text(luaL_checkstring(L, 1)), luaL_optinteger(L, 2, 0)));
  case LoadFont:  // (one font here, the system's, with every script's glyphs)
    lua_pushnil(L);
    return 1;
  case PushFont:
  case PopFont:
  case PushFontSize:
  case PopFontSize:
    return 0;
  case DefaultFontSize: return push_number(ImGui::GetFontSize());
  case GetMouse: return push_vec2(ImGui::GetMousePos());
  case ProgressBar:
    ImGui::ProgressBar(luaL_checknumber(L, 1), lua_isnoneornil(L, 2) ? ImVec2{-FLT_MIN, 0} : vec2(L, 2),
                       lua_isnoneornil(L, 3) ? nullptr : visible_text(luaL_checkstring(L, 3)));
    return 0;
  case KeyIndex: lua_pushinteger(L, luaL_checkinteger(L, 1)); return 1;
  case KeyDown: return push_bool(key_state(L, 0));
  case KeyPressed: return push_bool(key_state(L, 1));
  case KeyReleased: return push_bool(key_state(L, 2));
  case MouseClicked:
  case MouseReleased:
  case MouseDoubleClicked: {
    int button = luaL_optinteger(L, 1, 0);
    if (button < 0 || button >= ImGuiMouseButton_COUNT)
      return luaL_argerror(L, 1, "invalid mouse button");
    return push_bool(kind == MouseClicked ? ImGui::IsMouseClicked(button)
                     : kind == MouseReleased ? ImGui::IsMouseReleased(button)
                                             : ImGui::IsMouseDoubleClicked(button));
  }
  case SetTooltip: ImGui::SetTooltip("%s", visible_text(luaL_checkstring(L, 1))); return 0;
  case NextItemOpen: ImGui::SetNextItemOpen(lua_toboolean(L, 1), luaL_optinteger(L, 2, 0)); return 0;
  case BeginListBox: {
    bool open = ImGui::BeginListBox(visible_text(luaL_checkstring(L, 1)), lua_isnoneornil(L, 2) ? ImVec2{} : vec2(L, 2));
    if (open) scopes.push_back(Scope::ListBox);
    return push_bool(open);
  }
  case EndListBox: return end(Scope::ListBox);
  case BeginMenuBar:
  case BeginMainMenuBar: {
    bool open = kind == BeginMenuBar ? ImGui::BeginMenuBar() : ImGui::BeginMainMenuBar();
    if (open) scopes.push_back(kind == BeginMenuBar ? Scope::MenuBar : Scope::MainMenuBar);
    return push_bool(open);
  }
  case EndMenuBar: return end(Scope::MenuBar);
  case EndMainMenuBar: return end(Scope::MainMenuBar);
  case BeginMenu: {
    bool open = ImGui::BeginMenu(visible_text(luaL_checkstring(L, 1)), lua_isnoneornil(L, 2) || lua_toboolean(L, 2));
    if (open) scopes.push_back(Scope::Menu);
    return push_bool(open);
  }
  case EndMenu: return end(Scope::Menu);
  case MenuItem:
    return push_bool(ImGui::MenuItem(visible_text(luaL_checkstring(L, 1)), luaL_optstring(L, 2, nullptr),
                                     lua_toboolean(L, 3), lua_isnoneornil(L, 4) || lua_toboolean(L, 4)));
  case PushStyleColor: {
    int index = luaL_checkinteger(L, 1);
    if (index < 0 || index >= ImGuiCol_COUNT)
      return luaL_argerror(L, 1, "unknown style color");
    if (lua_isinteger(L, 2))
      ImGui::PushStyleColor(index, static_cast<ImU32>(lua_tointeger(L, 2)));
    else {
      float v[4]{};
      vecn(L, 2, v, 4);
      ImGui::PushStyleColor(index, ImVec4{v[0], v[1], v[2], v[3]});
    }
    scopes.push_back(Scope::StyleColor);
    return 0;
  }
  case PopStyleColor: {
    int count = luaL_optinteger(L, 1, 1);
    if (count < 0 || count > std::count(scopes.begin(), scopes.end(), Scope::StyleColor))
      return luaL_error(L, "pop_style_color exceeds the number pushed");
    ImGui::PopStyleColor(count);
    while (count--) {
      auto it = std::find(scopes.rbegin(), scopes.rend(), Scope::StyleColor);
      scopes.erase(std::next(it).base());
    }
    return 0;
  }
  case CursorStartPos: return push_vec2(ImGui::GetCursorStartPos());
  case SetItemDefaultFocus: ImGui::SetItemDefaultFocus(); return 0;
  case ScrollX: return push_number(ImGui::GetScrollX());
  case ScrollY: return push_number(ImGui::GetScrollY());
  case ScrollMaxX: return push_number(ImGui::GetScrollMaxX());
  case ScrollMaxY: return push_number(ImGui::GetScrollMaxY());
  case SetScrollX: ImGui::SetScrollX(luaL_checknumber(L, 1)); return 0;
  case SetScrollY: ImGui::SetScrollY(luaL_checknumber(L, 1)); return 0;
  case SetScrollHereX: ImGui::SetScrollHereX(luaL_optnumber(L, 1, .5)); return 0;
  case SetScrollHereY: ImGui::SetScrollHereY(luaL_optnumber(L, 1, .5)); return 0;
  case SetScrollFromPosX: ImGui::SetScrollFromPosX(luaL_checknumber(L, 1), luaL_optnumber(L, 2, .5)); return 0;
  case SetScrollFromPosY: ImGui::SetScrollFromPosY(luaL_checknumber(L, 1), luaL_optnumber(L, 2, .5)); return 0;
  case BeginTable: {
    int columns = luaL_checkinteger(L, 2);
    if (columns < 1 || columns > 512)
      return luaL_argerror(L, 2, "column count out of range");
    bool open = ImGui::BeginTable(luaL_checkstring(L, 1), columns, luaL_optinteger(L, 3, 0),
                                  lua_isnoneornil(L, 4) ? ImVec2{} : vec2(L, 4), luaL_optnumber(L, 5, 0));
    if (open) scopes.push_back(Scope::Table);
    return push_bool(open);
  }
  case EndTable: return end(Scope::Table);
  default:
    break;
  }
  // The rest only make sense inside a table
  if (std::find(scopes.begin(), scopes.end(), Scope::Table) == scopes.end()) {
    switch (kind) {
    case TableNextColumn:
    case TableSetColumnIndex: return push_bool(false);
    case TableColumnCount:
    case TableColumnIndex:
    case TableRowIndex:
    case TableColumnFlags: lua_pushinteger(L, 0); return 1;
    case TableColumnName: lua_pushliteral(L, ""); return 1;
    case SetClipboard:
    case GetClipboard: break;
    default: return 0;
    }
  }
  switch (kind) {
  case TableNextRow: ImGui::TableNextRow(luaL_optinteger(L, 1, 0), luaL_optnumber(L, 2, 0)); return 0;
  case TableNextColumn: return push_bool(ImGui::TableNextColumn());
  case TableSetColumnIndex: return push_bool(ImGui::TableSetColumnIndex(luaL_checkinteger(L, 1)));
  case TableSetupColumn:
    ImGui::TableSetupColumn(visible_text(luaL_checkstring(L, 1)), luaL_optinteger(L, 2, 0), luaL_optnumber(L, 3, 0),
                            static_cast<ImGuiID>(luaL_optinteger(L, 4, 0)));
    return 0;
  case TableSetupScrollFreeze: ImGui::TableSetupScrollFreeze(luaL_checkinteger(L, 1), luaL_checkinteger(L, 2)); return 0;
  case TableHeadersRow: ImGui::TableHeadersRow(); return 0;
  case TableHeader: ImGui::TableHeader(visible_text(luaL_checkstring(L, 1))); return 0;
  case TableColumnCount: lua_pushinteger(L, ImGui::TableGetColumnCount()); return 1;
  case TableColumnIndex: lua_pushinteger(L, ImGui::TableGetColumnIndex()); return 1;
  case TableRowIndex: lua_pushinteger(L, ImGui::TableGetRowIndex()); return 1;
  case TableColumnName: lua_pushstring(L, ImGui::TableGetColumnName(luaL_optinteger(L, 1, -1))); return 1;
  case TableColumnFlags: lua_pushinteger(L, ImGui::TableGetColumnFlags(luaL_optinteger(L, 1, -1))); return 1;
  case TableSetBgColor:
    ImGui::TableSetBgColor(luaL_checkinteger(L, 1), static_cast<ImU32>(luaL_checkinteger(L, 2)), luaL_optinteger(L, 3, -1));
    return 0;
  case SetClipboard: ImGui::SetClipboardText(luaL_checkstring(L, 1)); return 0;
  case GetClipboard: {
    const char *text = ImGui::GetClipboardText();
    lua_pushstring(L, text ? text : "");
    return 1;
  }
  default:
    return luaL_error(L, "Invalid ImGui widget");
  }
}
std::unique_ptr<ModFontGlyphs> glyphs;
GLint max_texture_size = 0;
std::chrono::steady_clock::time_point last;

// The settings menu's look (SettingsMenu.java: PANEL, ROW, EDGE, GOLD, CREAM, DIM, RED, GREEN): dark wine panels with a
// bronze edge, cream text, gold for what's on, the game's red for what's pressed. Scripts can still push their own.
ImVec4 rgb(uint32_t c, float a = 1) {
  return {float(c >> 16 & 255) / 255, float(c >> 8 & 255) / 255, float(c & 255) / 255, a};
}
void apply_theme(ImGuiStyle &st) {
  constexpr uint32_t panel = 0x201010, row = 0x2C1614, edge = 0xC49646, gold = 0xE2B65C, cream = 0xF0E4C8,
                     dim = 0xB09C84, red = 0xB0221E, wine = 0x5A1A16, wine_light = 0x7A2620;
  ImVec4 *c = st.Colors;
  c[ImGuiCol_Text] = rgb(cream);
  c[ImGuiCol_TextDisabled] = rgb(dim);
  c[ImGuiCol_WindowBg] = rgb(panel, .94f);
  c[ImGuiCol_ChildBg] = rgb(panel, 0);
  c[ImGuiCol_PopupBg] = rgb(panel, .97f);
  c[ImGuiCol_Border] = rgb(edge, .7f);
  c[ImGuiCol_BorderShadow] = rgb(0, 0);
  c[ImGuiCol_FrameBg] = rgb(row);
  c[ImGuiCol_FrameBgHovered] = rgb(wine);
  c[ImGuiCol_FrameBgActive] = rgb(wine_light);
  c[ImGuiCol_TitleBg] = rgb(row);
  c[ImGuiCol_TitleBgActive] = rgb(wine);
  c[ImGuiCol_TitleBgCollapsed] = rgb(row, .8f);
  c[ImGuiCol_MenuBarBg] = rgb(row);
  c[ImGuiCol_ScrollbarBg] = rgb(panel, .6f);
  c[ImGuiCol_ScrollbarGrab] = rgb(edge, .45f);
  c[ImGuiCol_ScrollbarGrabHovered] = rgb(edge, .7f);
  c[ImGuiCol_ScrollbarGrabActive] = rgb(gold);
  c[ImGuiCol_CheckMark] = rgb(gold);
  c[ImGuiCol_SliderGrab] = rgb(edge);
  c[ImGuiCol_SliderGrabActive] = rgb(gold);
  c[ImGuiCol_Button] = rgb(wine);
  c[ImGuiCol_ButtonHovered] = rgb(wine_light);
  c[ImGuiCol_ButtonActive] = rgb(red);
  c[ImGuiCol_Header] = rgb(wine);
  c[ImGuiCol_HeaderHovered] = rgb(wine_light);
  c[ImGuiCol_HeaderActive] = rgb(red);
  c[ImGuiCol_Separator] = rgb(edge, .5f);
  c[ImGuiCol_SeparatorHovered] = rgb(edge, .8f);
  c[ImGuiCol_SeparatorActive] = rgb(gold);
  c[ImGuiCol_ResizeGrip] = rgb(edge, .3f);
  c[ImGuiCol_ResizeGripHovered] = rgb(edge, .6f);
  c[ImGuiCol_ResizeGripActive] = rgb(gold);
  c[ImGuiCol_Tab] = rgb(row);
  c[ImGuiCol_TabHovered] = rgb(wine_light);
  c[ImGuiCol_TableHeaderBg] = rgb(row);
  c[ImGuiCol_TableBorderStrong] = rgb(edge, .6f);
  c[ImGuiCol_TableBorderLight] = rgb(edge, .3f);
  c[ImGuiCol_TableRowBg] = rgb(0, 0);
  c[ImGuiCol_TableRowBgAlt] = rgb(cream, .04f);
  c[ImGuiCol_TextSelectedBg] = rgb(red, .5f);
  c[ImGuiCol_PlotLines] = c[ImGuiCol_PlotHistogram] = rgb(gold);
  c[ImGuiCol_DragDropTarget] = rgb(gold);
  c[ImGuiCol_ModalWindowDimBg] = rgb(0, .5f);
  st.WindowRounding = 6;
  st.ChildRounding = st.FrameRounding = st.PopupRounding = st.GrabRounding = st.ScrollbarRounding = 4;
  st.WindowBorderSize = 2;
  st.FrameBorderSize = 1;  // (the settings menu's thin edge: a box is visible ticked or not)
  st.WindowPadding = {14, 12};
  st.FramePadding = {10, 9};  // (rows about as tall as a finger, like the settings menu's)
  st.ItemSpacing = {10, 10};
  st.ItemInnerSpacing = {10, 6};
  st.ScrollbarSize = 22;  // (a finger drags it)
  st.GrabMinSize = 22;
  st.WindowTitleAlign = {.5f, .5f};
}

// The style and text at a scale (the display's density times the menu size, SettingsMenu.uiScale): the font is made at
// the first one, so later sizes scale it
float font_scale = 1;   // what the font atlas was made at (0: ImGui's own small font, scaled entirely)
void restyle(float scale) {
  ImGuiStyle &st = ImGui::GetStyle();
  st = ImGuiStyle();
  apply_theme(st);
  st.ScaleAllSizes(scale);
  ImGui::GetIO().FontGlobalScale = font_scale > 0 ? scale / font_scale : scale * 18 / 13;  // (its font is 13 px)
}
float ui_scale = 1;
std::mutex errors_mutex;
std::vector<std::string> errors;  // (newest last, each once)
bool place_again = true;
std::atomic<float> window_alpha = .94f;  // (Mods → MOD UI opacity)
float content_height = 0;  // how tall last frame's contents were: the window fits them
int sections = 0, sections_drawn = 0;  // scripts with controls (last frame's, this frame's)  // (a new size puts the window back where a fresh one goes)

enum class Primitive { Text, Line, FilledRect, OutlineRect, FilledCircle, OutlineCircle, FilledQuad, OutlineQuad,
                       WorldToScreen, WorldText };
std::optional<rise::CameraMatrices> frame_camera;
int primitive(lua_State *L) {
  auto kind = static_cast<Primitive>(lua_tointeger(L, lua_upvalueindex(1)));
  if (!drawing && kind != Primitive::WorldToScreen)
    return luaL_error(L, "draw requires an active frame");
  if (kind == Primitive::WorldToScreen || kind == Primitive::WorldText) {
    bool text = kind == Primitive::WorldText;
    float xyz[3]{};
    if (ui && (mod_lua::read_components(L, text ? 2 : 1, false, xyz, 3) ||
               mod_lua::read_components(L, text ? 2 : 1, false, xyz, 2))) {
      try {
        auto size = ImGui::GetIO().DisplaySize;
        if (size.x > 0 && size.y > 0) {
          if (!drawing || !frame_camera) frame_camera = rise::camera_matrices(L);
          if (frame_camera) {
            auto xy = frame_camera->screen(xyz, size.x, size.y);
            if (xy) {
              if (!text) {
                mod_lua::push_components(L, false, xy->data(), 2);
                return 1;
              }
              ImGui::GetBackgroundDrawList()->AddText({(*xy)[0], (*xy)[1]},
                  luaL_optinteger(L, 3, IM_COL32_WHITE), visible_text(luaL_checkstring(L, 1)));
              return 0;
            }
          }
        }
      } catch (const std::exception &e) {
        lua_pushstring(L, e.what());
        return lua_error(L);
      }
    }
    if (text) return 0;
    lua_pushnil(L);
    return 1;
  }
  auto *list = ImGui::GetBackgroundDrawList();
  auto point = [&](int i) {
    return ImVec2{float(luaL_checknumber(L, i)), float(luaL_checknumber(L, i + 1))};
  };
  if (kind == Primitive::Text) {
    list->AddText(point(2), luaL_checkinteger(L, 4), visible_text(luaL_checkstring(L, 1)));
    return 0;
  }
  auto a = point(1);
  if (kind == Primitive::FilledCircle || kind == Primitive::OutlineCircle) {
    float radius = luaL_checknumber(L, 3);
    ImU32 color = luaL_checkinteger(L, 4);
    int segments = luaL_optinteger(L, 5, 32);
    if (kind == Primitive::FilledCircle) list->AddCircleFilled(a, radius, color, segments);
    else list->AddCircle(a, radius, color, segments);
    return 0;
  }
  auto b = point(3);
  if (kind == Primitive::FilledQuad || kind == Primitive::OutlineQuad) {
    auto c = point(5), d = point(7);
    ImU32 color = luaL_checkinteger(L, 9);
    if (kind == Primitive::FilledQuad) list->AddQuadFilled(a, b, c, d, color);
    else list->AddQuad(a, b, c, d, color);
    return 0;
  }
  ImU32 color = luaL_checkinteger(L, 5);
  if (kind == Primitive::Line) list->AddLine(a, b, color);
  else if (kind == Primitive::FilledRect) list->AddRectFilled(a, {a.x+b.x, a.y+b.y}, color);
  else list->AddRect(a, {a.x+b.x, a.y+b.y}, color);
  return 0;
}
} // namespace
void mod_lua::script_error(const std::string &message) {
  std::lock_guard lock(errors_mutex);
  if (std::find(errors.begin(), errors.end(), message) != errors.end()) return;
  if (errors.size() == 20) errors.erase(errors.begin());
  errors.push_back(message);
}
void mod_lua::script_error_over(const std::string &message) {
  std::lock_guard lock(errors_mutex);
  errors.erase(std::remove(errors.begin(), errors.end(), message), errors.end());
}
void mod_lua::clear_script_errors() {
  std::lock_guard lock(errors_mutex);
  errors.clear();
}
static void draw_errors() {
  std::lock_guard lock(errors_mutex);
  if (errors.empty()) return;
  ImGui::PushStyleColor(ImGuiCol_Text, rgb(0xF08070));
  bool open = ImGui::CollapsingHeader(("Script errors (" + std::to_string(errors.size()) + ")").c_str(),
                                      ImGuiTreeNodeFlags_DefaultOpen);
  ImGui::PopStyleColor();
  if (!open) return;
  ImGui::TextDisabled("Errors still happening. A script that stops on one while loading adds nothing below.");
  for (auto &e : errors)
    ImGui::TextWrapped("%s", e.c_str());
  ImGui::Separator();
}
int mod_lua::call_ui_callback(lua_State *L, const char *id) {
  auto base = scopes.size();
  if (id && drawing) ImGui::PushID(id);  // (two scripts can both have a "Enabled" checkbox)
  int status = lua_pcall(L, 0, 0, 0);
  while (scopes.size() > base)
    close_scope();
  if (id && drawing) ImGui::PopID();
  return status;
}
bool mod_lua::script_section(const char *name) {
  if (!drawing) return true;
  ++sections_drawn;
  ImGui::PushStyleColor(ImGuiCol_Text, rgb(0xE2B65C));
  bool open = ImGui::CollapsingHeader(visible_text(name), ImGuiTreeNodeFlags_DefaultOpen);
  ImGui::PopStyleColor();
  return open;
}
bool mod_lua::ui_visible() { return visible.load(); }
void mod_lua::service_ui_frame() {
  frame_camera.reset();
  // BeginRendering owns script execution. The GLES thread consumes only completed
  // draw data, under the same mutex, and may skip superseded game frames.
  if (ui) ImGui::SetCurrentContext(ui);
  if (!ui || !ImGui::GetIO().Fonts->IsBuilt() ||
      ImGui::GetIO().DisplaySize.x <= 0 || ImGui::GetIO().DisplaySize.y <= 0) {
    draw_frame();
    return;
  }
  auto now = std::chrono::steady_clock::now();
  auto &io = ImGui::GetIO();
  io.DeltaTime = std::max(0.0001f, std::chrono::duration<float>(now - last).count());
  last = now;
  ImGui::NewFrame();
  drawing = true;
  if (visible) {
    // As wide as the settings menu (940 dp at most, 94% of the screen), from near the top, and only as tall as what
    // the scripts draw (up to the screen's bottom edge, then it scrolls)
    auto cond = place_again ? ImGuiCond_Always : ImGuiCond_FirstUseEver;
    place_again = false;
    const float w = std::min(io.DisplaySize.x * .94f, 940 * ui_scale), top = io.DisplaySize.y * .04f;
    ImGui::SetNextWindowPos({io.DisplaySize.x * .5f, top}, cond, {.5f, 0});
    float y = top;
    if (auto *win = ImGui::FindWindowByName("###MOD UI")) y = win->Pos.y;
    const float most = std::max(io.DisplaySize.y * .3f, io.DisplaySize.y * .98f - y);
    ImGui::SetNextWindowSize({w, std::clamp(content_height, std::min(160 * ui_scale, most), most)});
    ImGui::SetNextWindowBgAlpha(window_alpha);
    // The settings menu's title: gold, with how many mods have controls
    std::string title = "MOD UI";
    if (sections) title += "  \u25C6  " + std::to_string(sections) + (sections == 1 ? " mod" : " mods");
    title += "###MOD UI";
    bool open = true;
    ImGui::PushStyleColor(ImGuiCol_Text, rgb(0xE2B65C));
    bool shown = ImGui::Begin(visible_text(title.c_str()), &open, ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoResize);
    ImGui::PopStyleColor();
    sections_drawn = 0;
    if (shown) {
      draw_errors();
      draw_ui();
      content_height = ImGui::GetCursorPosY() - ImGui::GetStyle().ItemSpacing.y + ImGui::GetStyle().WindowPadding.y;
      sections = sections_drawn;
    }
    ImGui::End();
    if (!open) visible = false;
  }
  draw_frame();
  drawing = false;
  ImGui::Render();
  bool geometry = ImGui::GetDrawData()->TotalVtxCount > 0;
  if (geometry || had_geometry) needs_render = true;
  had_geometry = geometry;
}
void mod_lua::open_imgui(lua_State *L) {
  lua_newtable(L);
  for (auto [name, kind] : {std::pair{"text", Text},
                            {"text_colored", TextColored},
                            {"same_line", SameLine},
                            {"spacing", Spacing},
                            {"separator", Separator},
                            {"tree_node", Tree},
                            {"tree_pop", Pop},
                            {"button", Button},
                            {"checkbox", Check},
                            {"drag_int", DragInt},
                            {"slider_int", SliderInt},
                            {"drag_float", DragFloat},
                            {"slider_float", Slider},
                            {"input_text", Input},
                            {"new_line", NewLine},
                            {"is_item_hovered", Hovered},
                            {"begin_tooltip", Tooltip},
                            {"end_tooltip", EndTooltip},
                            {"begin_window", BeginWindow},
                            {"end_window", EndWindow},
                            {"set_next_window_pos", NextPos},
                            {"set_next_window_size", NextSize},
                            {"get_display_size", DisplaySize},
                            {"get_window_pos", WindowPos},
                            {"is_mouse_down", MouseDown},
                            {"push_id", PushId},
                            {"pop_id", PopId},
                            {"push_style_var", PushStyle},
                            {"pop_style_var", PopStyle},
                            {"indent", Indent},
                            {"unindent", Unindent},
                            {"combo", Combo}, {"calc_text_size", CalcTextSize},
                            {"get_window_size", WindowSize}, {"get_cursor_pos", CursorPos},
                            {"get_cursor_screen_pos", CursorScreenPos}, {"set_cursor_pos", SetCursorPos},
                            {"set_cursor_screen_pos", SetCursorScreenPos}, {"is_item_active", ItemActive},
                            {"is_item_focused", ItemFocused}, {"set_next_item_width", SetNextItemWidth},
                            {"calc_item_width", CalcItemWidth}, {"push_item_width", PushItemWidth},
                            {"pop_item_width", PopItemWidth}, {"begin_group", BeginGroup}, {"end_group", EndGroup},
                            {"begin_rect", BeginRect}, {"end_rect", EndRect}, {"begin_disabled", BeginDisabled},
                            {"end_disabled", EndDisabled}, {"open_popup", OpenPopup}, {"begin_popup", BeginPopup},
                            {"begin_popup_context_item", BeginPopupContextItem}, {"end_popup", EndPopup},
                            {"close_current_popup", CloseCurrentPopup}, {"is_popup_open", IsPopupOpen},
                            {"input_text_multiline", InputMultiline}, {"color_picker_argb", ColorPickerArgb},
                            {"begin_child_window", BeginChild}, {"end_child_window", EndChild},
                            {"small_button", SmallButton}, {"invisible_button", InvisibleButton},
                            {"arrow_button", ArrowButton}, {"drag_float2", DragFloat2}, {"drag_float3", DragFloat3},
                            {"drag_float4", DragFloat4}, {"color_picker", ColorPicker}, {"color_picker3", ColorPicker3},
                            {"color_picker4", ColorPicker4}, {"color_edit", ColorEdit}, {"color_edit_argb", ColorEditArgb},
                            {"color_edit3", ColorEdit3}, {"color_edit4", ColorEdit4}, {"tree_node_ptr_id", TreeNodePtr},
                            {"tree_node_str_id", TreeNodeStr}, {"collapsing_header", CollapsingHeader},
                            {"load_font", LoadFont}, {"push_font", PushFont}, {"pop_font", PopFont},
                            {"get_default_font_size", DefaultFontSize}, {"get_mouse", GetMouse},
                            {"progress_bar", ProgressBar}, {"get_key_index", KeyIndex}, {"is_key_down", KeyDown},
                            {"is_key_pressed", KeyPressed}, {"is_key_released", KeyReleased},
                            {"is_mouse_clicked", MouseClicked}, {"is_mouse_released", MouseReleased},
                            {"is_mouse_double_clicked", MouseDoubleClicked}, {"set_tooltip", SetTooltip},
                            {"set_next_item_open", NextItemOpen}, {"begin_list_box", BeginListBox},
                            {"end_list_box", EndListBox}, {"begin_menu_bar", BeginMenuBar}, {"end_menu_bar", EndMenuBar},
                            {"begin_main_menu_bar", BeginMainMenuBar}, {"end_main_menu_bar", EndMainMenuBar},
                            {"begin_menu", BeginMenu}, {"end_menu", EndMenu}, {"menu_item", MenuItem},
                            {"push_style_color", PushStyleColor}, {"pop_style_color", PopStyleColor},
                            {"get_cursor_start_pos", CursorStartPos}, {"set_item_default_focus", SetItemDefaultFocus},
                            {"get_scroll_x", ScrollX}, {"get_scroll_y", ScrollY}, {"set_scroll_x", SetScrollX},
                            {"set_scroll_y", SetScrollY}, {"get_scroll_max_x", ScrollMaxX},
                            {"get_scroll_max_y", ScrollMaxY}, {"set_scroll_here_x", SetScrollHereX},
                            {"set_scroll_here_y", SetScrollHereY}, {"set_scroll_from_pos_x", SetScrollFromPosX},
                            {"set_scroll_from_pos_y", SetScrollFromPosY}, {"begin_table", BeginTable},
                            {"end_table", EndTable}, {"table_next_row", TableNextRow},
                            {"table_next_column", TableNextColumn}, {"table_set_column_index", TableSetColumnIndex},
                            {"table_setup_column", TableSetupColumn}, {"table_setup_scroll_freeze", TableSetupScrollFreeze},
                            {"table_headers_row", TableHeadersRow}, {"table_header", TableHeader},
                            {"table_get_column_count", TableColumnCount}, {"table_get_column_index", TableColumnIndex},
                            {"table_get_row_index", TableRowIndex}, {"table_get_column_name", TableColumnName},
                            {"table_get_column_flags", TableColumnFlags}, {"table_set_bg_color", TableSetBgColor},
                            {"set_clipboard", SetClipboard}, {"get_clipboard", GetClipboard},
                            {"push_font_size", PushFontSize}, {"pop_font_size", PopFontSize}}) {
    lua_pushinteger(L, kind);
    lua_pushcclosure(L, widget, 1);
    lua_setfield(L, -2, name);
  }
  lua_setglobal(L, "imgui");
  lua_newtable(L);
  for (auto [name, kind] : {std::pair{"text", Primitive::Text}, {"line", Primitive::Line},
                            {"filled_rect", Primitive::FilledRect}, {"outline_rect", Primitive::OutlineRect},
                            {"filled_circle", Primitive::FilledCircle}, {"outline_circle", Primitive::OutlineCircle},
                            {"filled_quad", Primitive::FilledQuad}, {"outline_quad", Primitive::OutlineQuad},
                            {"world_to_screen", Primitive::WorldToScreen}, {"world_text", Primitive::WorldText}}) {
    lua_pushinteger(L, static_cast<int>(kind));
    lua_pushcclosure(L, primitive, 1);
    lua_setfield(L, -2, name);
  }
  lua_setglobal(L, "draw");
}
extern "C" JNIEXPORT void JNICALL
Java_io_github_iamwilmar_mhriseport_ModUi_nativeCreate(JNIEnv *, jclass,
                                                       jfloat scale) {
  std::lock_guard lock(mod_lua::lua_mutex());
  ui = ImGui::CreateContext();
  ImGui::SetCurrentContext(ui);
  auto &io = ImGui::GetIO();
  io.IniFilename = nullptr;
  ui_scale = scale;
  // (18 at 1: between the settings menu's 16 sp rows and its titles, since this text is denser)
  const char *font = "/system/fonts/NotoSansCJK-Regular.ttc";
  font_scale = 0;
  if (auto f = fopen(font, "rb")) {
    fclose(f);
    glyphs = std::make_unique<ModFontGlyphs>(*io.Fonts);
    if (io.Fonts->AddFontFromFileTTF(font, 18 * scale, nullptr, glyphs->ranges())) {
      io.UserData = glyphs.get();
      font_scale = scale;
    } else
      glyphs.reset();
  } else {
    io.Fonts->AddFontDefault();
  }
  if (!io.Fonts->Build()) {
    __android_log_write(ANDROID_LOG_ERROR, "RiseLua",
                        "System font atlas failed to build");
    io.Fonts->Clear();
    io.UserData = nullptr;
    glyphs.reset();
    io.Fonts->AddFontDefault();
    font_scale = 0;
    if (!io.Fonts->Build()) {
      ImGui::DestroyContext(ui);
      ui = nullptr;
      return;
    }
  }
  restyle(scale);
  ImGui_ImplOpenGL3_Init("#version 300 es");
  ImGui_ImplOpenGL3_NewFrame();
  glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_texture_size);
  last = std::chrono::steady_clock::now();
  needs_render = true;
  had_geometry = false;
}
// Mods → MOD UI opacity
extern "C" JNIEXPORT void JNICALL
Java_io_github_iamwilmar_mhriseport_ModUi_nativeOpacity(JNIEnv *, jclass, jfloat alpha) {
  window_alpha = std::clamp(float(alpha), .3f, 1.f);
}

// Settings → Display → Menu size changed (or the screen did): the same font, scaled
extern "C" JNIEXPORT void JNICALL
Java_io_github_iamwilmar_mhriseport_ModUi_nativeScale(JNIEnv *, jclass, jfloat scale) {
  std::lock_guard lock(mod_lua::lua_mutex());
  if (!ui || scale <= 0 || scale == ui_scale) return;
  ui_scale = scale;
  place_again = true;
  restyle(scale);
  needs_render = true;
}
extern "C" JNIEXPORT void JNICALL
Java_io_github_iamwilmar_mhriseport_ModUi_nativeSize(JNIEnv *, jclass, jint w, jint h) {
  std::lock_guard lock(mod_lua::lua_mutex());
  if (ui) ImGui::GetIO().DisplaySize = ImVec2(w, h);
  needs_render = true;
}
extern "C" JNIEXPORT jboolean JNICALL
Java_io_github_iamwilmar_mhriseport_ModUi_nativePending(JNIEnv *, jclass) {
  return needs_render.load();
}
extern "C" JNIEXPORT void JNICALL
Java_io_github_iamwilmar_mhriseport_ModUi_nativeMenu(JNIEnv *, jclass, jboolean open) {
  visible = open;
}
extern "C" JNIEXPORT jboolean JNICALL
Java_io_github_iamwilmar_mhriseport_ModUi_nativeMenuOpen(JNIEnv *, jclass) {
  return visible.load();
}
extern "C" JNIEXPORT jint JNICALL
Java_io_github_iamwilmar_mhriseport_ModUi_nativeFrame(JNIEnv *, jclass, jint w,
                                                      jint h) {
  std::lock_guard lock(mod_lua::lua_mutex());
  needs_render = false;
  glViewport(0, 0, w, h);
  glClearColor(0, 0, 0, 0);
  glClear(GL_COLOR_BUFFER_BIT);
  if (!ui || w <= 0 || h <= 0)
    return 0;
  ImGui::SetCurrentContext(ui);
  auto &io = ImGui::GetIO();
  if (auto *data = ImGui::GetDrawData())
    ImGui_ImplOpenGL3_RenderDrawData(data);
  if (glyphs) {
    if (!io.Fonts->IsBuilt()) return 0;
    auto update = glyphs->update(*io.Fonts, max_texture_size);
    if (update != ModFontGlyphs::Update::Unchanged) {
      if (update != ModFontGlyphs::Update::Built)
        __android_log_write(ANDROID_LOG_ERROR, "RiseLua", "Could not expand MOD font atlas");
      if (update == ModFontGlyphs::Update::Failed) return 0;
      ImGui_ImplOpenGL3_DestroyFontsTexture();
      ImGui_ImplOpenGL3_CreateFontsTexture();
    }
  }
  return (visible ? 1 : 0) | (io.WantTextInput ? 2 : 0);
}
extern "C" JNIEXPORT jboolean JNICALL
Java_io_github_iamwilmar_mhriseport_ModUi_nativeWindowAt(JNIEnv *, jclass, jfloat x, jfloat y) {
  std::lock_guard lock(mod_lua::lua_mutex());
  if (!ui || ui->FrameCount == 0) return false;
  ImGui::SetCurrentContext(ui);
  ImGuiWindow *window = nullptr;
  // Hit-test the new contact, independently of any window another finger is moving.
  ImGui::FindHoveredWindowEx({x, y}, true, &window, nullptr);
  return window && window->Active && !window->IsFallbackWindow;
}
extern "C" JNIEXPORT void JNICALL
Java_io_github_iamwilmar_mhriseport_ModUi_nativeTouch(JNIEnv *, jclass,
                                                      jfloat x, jfloat y,
                                                      jboolean down) {
  std::lock_guard lock(mod_lua::lua_mutex());
  if (!ui)
    return;
  ImGui::SetCurrentContext(ui);
  auto &io = ImGui::GetIO();
  io.AddMouseSourceEvent(ImGuiMouseSource_TouchScreen);
  io.AddMousePosEvent(x, y);
  io.AddMouseButtonEvent(0, down);
}
extern "C" JNIEXPORT void JNICALL
Java_io_github_iamwilmar_mhriseport_ModUi_nativeChar(JNIEnv *, jclass,
                                                     jint cp) {
  std::lock_guard lock(mod_lua::lua_mutex());
  if (!ui)
    return;
  ImGui::SetCurrentContext(ui);
  auto &io = ImGui::GetIO();
  if (cp == 8) {
    io.AddKeyEvent(ImGuiKey_Backspace, true);
    io.AddKeyEvent(ImGuiKey_Backspace, false);
  } else if (cp > 0)
    io.AddInputCharacter(cp);
}
extern "C" JNIEXPORT void JNICALL
Java_io_github_iamwilmar_mhriseport_ModUi_nativeClose(JNIEnv *, jclass) {
  std::lock_guard lock(mod_lua::lua_mutex());
  if (!ui)
    return;
  ImGui::SetCurrentContext(ui);
  ImGui_ImplOpenGL3_Shutdown();
  ImGui::DestroyContext(ui);
  glyphs.reset();
  ui = nullptr;
  needs_render = false;
}
