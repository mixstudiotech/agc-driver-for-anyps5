#define NOMINMAX
#include "prx/libc/include/WindowsMappings.hpp"
#include <array>
#include <iostream>
#include <stdexcept>

namespace {
using Mappings = GuestArena::WindowsMappings;
constexpr std::size_t Bytes = 0x10000;
constexpr std::size_t Page = 0x4000;

void Require(bool value) {
    if (!value) throw std::runtime_error("physical alias identity check failed");
}

void* Import(unsigned char* guest, std::size_t bytes) {
    auto refused = Mappings::ImportRefusal::None;
    auto* alias = Mappings::Get().MapImportAlias(reinterpret_cast<std::uintptr_t>(guest), bytes, &refused);
    Require(alias != nullptr && refused == Mappings::ImportRefusal::None);
    return alias;
}
}

int main() {
    try {
        auto& mappings = Mappings::Get();
        const auto window = Mappings::importWindowBytes;
        auto* reserved = static_cast<unsigned char*>(mappings.Reserve(nullptr, window * 2));
        Require(reserved != nullptr);
        auto* guest = reinterpret_cast<unsigned char*>((reinterpret_cast<std::uintptr_t>(reserved) + window - 1) & ~(window - 1));
        HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE | SEC_RESERVE, 0, Bytes, nullptr);
        Require(section != nullptr);
        mappings.Map(guest, Bytes, section, 0, PAGE_READWRITE, true, true);
        auto* first = Import(guest, Bytes);
        std::array<std::uint64_t, 4> before{};
        Require(mappings.ImportAliasPages(first, before.data(), before.size()) == before.size());
        for (std::size_t page = 0; page < before.size(); ++page) {
            Require(before[page] != 0);
            for (std::size_t other = 0; other < page; ++other) Require(before[page] != before[other]);
        }
        mappings.Reset(guest, Bytes);
        mappings.Map(guest + Bytes, Bytes, section, 0, PAGE_READWRITE, true, true);
        auto* second = Import(guest + Bytes, Bytes);
        std::array<std::uint64_t, 4> after{};
        Require(mappings.ImportAliasPages(second, after.data(), after.size()) == after.size());
        Require(before == after);
        auto* part = Import(guest + Bytes + Page, Page * 2);
        std::array<std::uint64_t, 2> middle{};
        Require(mappings.ImportAliasPages(part, middle.data(), middle.size()) == middle.size());
        Require(middle[0] == before[1] && middle[1] == before[2]);
        mappings.UnmapAlias(part);
        mappings.UnmapAlias(second);
        mappings.UnmapAlias(first);
        mappings.Reset(reserved, window * 2);
        Require(VirtualFree(reserved, 0, MEM_RELEASE));
        mappings.ForgetSection(section);
        CloseHandle(section);
        std::cout << "AMM physical page identities survive remap and preserve offsets\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
