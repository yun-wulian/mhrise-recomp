#include "loader/lua_explorer.h"
#include "loader/lua_mod_hooks.h"
#include "loader/tdb71.h"
#include "nnsdk/ini.h"
#include <algorithm>
#include <android/log.h>
#include <cctype>
#include <lua.hpp>
#include <string>

namespace {
char reported_key;

int missing_call(lua_State *L) {
  if (lua_type(L, 2) != LUA_TSTRING)
    return 0;
  const char *key = lua_tostring(L, 2);
  std::string name = std::string(lua_tostring(L, lua_upvalueindex(1))) + "." + key;
  lua_rawgetp(L, LUA_REGISTRYINDEX, &reported_key);
  if (!lua_istable(L, -1)) {
    lua_pop(L, 1);
    lua_newtable(L);
    lua_pushvalue(L, -1);
    lua_rawsetp(L, LUA_REGISTRYINDEX, &reported_key);
  }
  lua_getfield(L, -1, name.c_str());
  if (lua_isnil(L, -1)) {
    lua_pushboolean(L, 1);
    lua_setfield(L, -3, name.c_str());
    std::string where = "a script";
    lua_Debug caller{};
    if (lua_getstack(L, 1, &caller) && lua_getinfo(L, "Sl", &caller)) {
      where = caller.short_src;
      if (caller.currentline > 0)
        where += ":" + std::to_string(caller.currentline);
    }
    mod_lua::log_message(ANDROID_LOG_WARN,
                         (name + " is unavailable (" + where + ")").c_str());
  }
  return 0;  // Feature detection must see nil for unsupported fields.
}

// Every type whose full name has the text (any case), up to a limit
int search_types(lua_State *L) {
  auto &t = *static_cast<rise::Tdb *>(lua_touserdata(L, lua_upvalueindex(1)));
  std::string want = luaL_checkstring(L, 1);
  std::transform(want.begin(), want.end(), want.begin(), [](unsigned char c) { return std::tolower(c); });
  const lua_Integer limit = luaL_optinteger(L, 2, 200);
  lua_newtable(L);
  lua_Integer found = 0;
  const size_t count = t.types.size() / 72;
  for (size_t id = 1; id < count && found < limit; ++id) {
    std::string name;
    try {
      name = t.full_name(static_cast<uint32_t>(id));
    } catch (const std::exception &) {
      continue;
    }
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
    if (lower.find(want) == std::string::npos)
      continue;
    lua_pushlstring(L, name.data(), name.size());
    lua_rawseti(L, -2, ++found);
  }
  return 1;
}

// The explorer itself, in Lua over the same sdk and imgui calls the scripts get
constexpr char explorer_source[] = R"lua(
local search, results, status = "", {}, ""
local opened, called = {}, {}
local LIMIT = 64
local common = {
  "snow.player.PlayerManager", "snow.enemy.EnemyManager", "snow.QuestManager", "snow.data.DataManager",
  "snow.gui.GuiManager", "snow.gui.MessageManager", "snow.CameraManager",
}

local function address(v)
  local ok, a = pcall(v.get_address, v)
  return ok and a or 0
end

local function managed(v)
  return type(v) == "userdata" and sdk.is_managed_object(v)
end

local function describe(v)
  if v == nil then return "nil" end
  if managed(v) then
    local ok, td = pcall(v.get_type_definition, v)
    return (ok and td) and td:get_full_name() or "object"
  end
  if math.type(v) == "float" then return string.format("%.4f", v) end
  return tostring(v)
end

local show_object

local function show_value(label, v, depth)
  if managed(v) then
    if imgui.tree_node_ptr_id(address(v), label .. ": " .. describe(v)) then
      show_object(v, depth + 1)
      imgui.tree_pop()
    end
  else
    imgui.text(label .. ": " .. describe(v))
  end
end

show_object = function(obj, depth)
  if depth > 12 then
    imgui.text("...")
    return
  end
  local ok, td = pcall(obj.get_type_definition, obj)
  if not ok or not td then
    imgui.text("(no type)")
    return
  end
  imgui.text(string.format("%s at 0x%X", td:get_full_name(), address(obj)))
  if td:is_array() then
    local got, size = pcall(obj.get_size, obj)
    if not got then return end
    imgui.text(size .. " elements")
    for i = 0, math.min(size, LIMIT) - 1 do
      local fine, item = pcall(obj.get_element, obj, i)
      if fine then show_value("[" .. i .. "]", item, depth) else imgui.text("[" .. i .. "]: " .. tostring(item)) end
    end
    if size > LIMIT then imgui.text((size - LIMIT) .. " more") end
    return
  end
  local level = td
  while level do
    local fields = level:get_fields()
    if #fields > 0 then
      imgui.text_colored(level:get_full_name(), 0xFF7FC8FF)
      for _, f in ipairs(fields) do
        if not f:is_static() then
          local fine, v = pcall(f.get_data, f, obj)
          if fine then show_value(f:get_name(), v, depth) else imgui.text(f:get_name() .. ": " .. tostring(v)) end
        end
      end
    end
    level = level:get_parent_type()
  end
  if imgui.tree_node_str_id("getters", "Getters") then
    local here = address(obj)
    for _, m in ipairs(td:get_methods()) do
      local name = m:get_name()
      if not m:is_static() and m:get_num_params() == 0 and name:sub(1, 4) == "get_" then
        local key = here .. name
        imgui.push_id(name)
        if imgui.small_button("call") then
          local fine, r = pcall(m.call, m, obj)
          called[key] = { value = fine and r or tostring(r) }
        end
        imgui.same_line()
        if called[key] then show_value(name, called[key].value, depth) else imgui.text(name) end
        imgui.pop_id()
      end
    end
    imgui.tree_pop()
  end
end

local function signature(m)
  local params = {}
  local ok, types = pcall(m.get_param_types, m)
  if ok then
    for i, p in ipairs(types) do params[i] = p and p:get_full_name() or "?" end
  end
  local ret = m:get_return_type()
  return string.format("%s%s(%s) : %s", m:is_static() and "static " or "", m:get_name(),
                       table.concat(params, ", "), ret and ret:get_full_name() or "void")
end

local function show_type(name)
  local td = sdk.find_type_definition(name)
  if not td then
    imgui.text("Not in this game")
    return
  end
  local parent = td:get_parent_type()
  imgui.text("Parent: " .. (parent and parent:get_full_name() or "none"))
  imgui.text(string.format("Size 0x%X%s", td:get_size(), td:is_value_type() and ", value type" or ""))
  local ok, getter = pcall(td.get_method, td, "get_Instance")
  if ok and getter then
    if imgui.button("Open its instance") then
      local fine, inst = pcall(sdk.get_managed_singleton, name)
      if fine and inst then
        opened[name] = inst
        status = ""
      else
        status = name .. " has no instance right now"
      end
    end
  end
  if imgui.tree_node_str_id("fields", "Fields") then
    for _, f in ipairs(td:get_fields()) do
      local ft = f:get_type()
      local fine, offset = pcall(f.get_offset_from_base, f)
      imgui.text(string.format("%s%s : %s%s", f:is_static() and "static " or "", f:get_name(),
                               ft and ft:get_full_name() or "?", fine and string.format("  (+0x%X)", offset) or ""))
    end
    imgui.tree_pop()
  end
  if imgui.tree_node_str_id("methods", "Methods") then
    for _, m in ipairs(td:get_methods()) do
      local fine, text = pcall(signature, m)
      imgui.text(fine and text or tostring(text))
    end
    imgui.tree_pop()
  end
end

re.on_draw_ui(function()
  if not imgui.collapsing_header("Object Explorer") then return end
  local changed, text = imgui.input_text("Type", search)
  if changed then search = text end
  imgui.same_line()
  if imgui.button("Search") then
    if #search < 3 then
      status = "Type at least 3 letters of a type name"
    else
      results = __explorer_search(search, 200)
      status = #results == 200 and "The first 200 types" or (#results .. " types")
    end
  end
  if status ~= "" then imgui.text(status) end
  for name, inst in pairs(opened) do
    if imgui.tree_node_str_id("instance " .. name, "Instance of " .. name) then
      if imgui.small_button("Close") then opened[name] = nil end
      show_object(inst, 0)
      imgui.tree_pop()
    end
  end
  if imgui.tree_node_str_id("common", "Managers") then
    for _, name in ipairs(common) do
      if imgui.tree_node_str_id("common " .. name, name) then
        show_type(name)
        imgui.tree_pop()
      end
    end
    imgui.tree_pop()
  end
  for _, name in ipairs(results) do
    if imgui.tree_node_str_id("type " .. name, name) then
      show_type(name)
      imgui.tree_pop()
    end
  end
end)
)lua";
} // namespace

void mod_lua::watch_missing_calls(lua_State *L) {
  for (const char *table : {"sdk", "re", "imgui", "draw", "log", "fs", "json", "reframework", "thread", "vrmod",
                            "imnodes", "imguizmo", "object_explorer"}) {
    lua_getglobal(L, table);
    if (!lua_istable(L, -1)) {
      lua_pop(L, 1);
      continue;
    }
    if (!lua_getmetatable(L, -1)) {
      lua_newtable(L);
      lua_pushstring(L, table);
      lua_pushcclosure(L, missing_call, 1);
      lua_setfield(L, -2, "__index");
      lua_setmetatable(L, -2);
    } else
      lua_pop(L, 1);
    lua_pop(L, 1);
  }
}

void mod_lua::load_object_explorer(lua_State *L, rise::Tdb &tdb) {
  if (!nnsdk::ini::on("object_explorer"))
    return;
  lua_pushlightuserdata(L, &tdb);
  lua_pushcclosure(L, search_types, 1);
  lua_setglobal(L, "__explorer_search");
  if (luaL_loadbufferx(L, explorer_source, sizeof(explorer_source) - 1, "=object_explorer", "t") != LUA_OK ||
      lua_pcall(L, 0, 0, 0) != LUA_OK) {
    const char *error = lua_tostring(L, -1);
    log_message(ANDROID_LOG_ERROR, (std::string("Object Explorer: ") + (error ? error : "error")).c_str());
    lua_pop(L, 1);
  }
}
