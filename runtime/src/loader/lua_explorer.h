#pragma once
struct lua_State;
namespace rise {
struct Tdb;
}
namespace mod_lua {
// A script reading a REFramework call this port doesn't have yet gets a function that does nothing, and the log and
// Debug → Lua / TDB say which call and which script, once each: so the script goes on, and it's clear what to add.
void watch_missing_calls(lua_State *);
// Debug → Object Explorer (object_explorer in mhr.ini): REFramework's explorer, in the MOD UI
void load_object_explorer(lua_State *, rise::Tdb &);
} // namespace mod_lua
