#pragma once
#include <string>
struct lua_State;
namespace mod_lua {
void open_imgui(lua_State *);
void service_ui_frame();
int call_ui_callback(lua_State *, const char *id = nullptr);
bool script_section(const char *name);
bool ui_visible();
// Shown at the top of the MOD UI, since a script that fails while loading draws nothing
void script_error(const std::string &message);
void clear_script_errors();
// One that stopped happening (a callback or hook that has run without it since)
void script_error_over(const std::string &message);
} // namespace mod_lua
