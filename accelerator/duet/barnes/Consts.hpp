#pragma once

// Duet Registers IDs
// Need to match these in the Gem5 Duet implementation
// Max 16 registers
constexpr auto kArg = 0;
constexpr auto kPos0X = 1;
constexpr auto kPos0Y = 2;
constexpr auto kPos0Z = 3;
constexpr auto kFincnt = 4;
constexpr auto kAccX = 5;
constexpr auto kAccY = 6;
constexpr auto kAccZ = 7;

// Consts
constexpr auto kNEngine = 1;
constexpr auto kDuetLeafSize = 32;
constexpr auto kEpssq = 1e-9;
