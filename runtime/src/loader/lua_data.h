#pragma once
#include <filesystem>
#include <vector>
struct lua_State;
namespace mod_lua {
std::vector<std::filesystem::path> framework_roots(const std::filesystem::path &private_root);
void open_data(lua_State *, const std::filesystem::path &, const std::vector<std::filesystem::path> &roots);
}
