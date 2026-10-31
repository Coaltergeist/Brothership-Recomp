// SHADER_PLAN Phase 4: the local store of translated stages. The seed (pipeline.cpp, Phase 5) translates every
// stage of the game files before play exactly as play does; the store keeps that translation - the SPIR-V,
// drippu's Shader::Info and the checks a use must pass - so the first play skips our translation too, not only
// the driver's compile (HEAVY_FRAMES_PLAN 2026-10-28: a cutscene's first heavy frames lost ~89 ms to 23 new
// pipelines whose compiles the driver's cache had). One file per game next to the pipeline cache
// (<RT_PIPELINE_CACHE or "cache">/<program id>.nvn.stages): a header with the tag (store version, drippu's commit,
// device, driver), then records {key, entry}; the index (key -> place) is read at start, entries on first use.
// RT_SHADER_STORE=0: not used (nor written). RT_SHADER_STORE_CHECK=1: a hit is also translated, the two compared
// (the fresh translation is used).
#pragma once

#include "pipeline.h"

#include <optional>
#include <string>
#include <vector>

namespace rt::nvn {

struct StoredStage {
    std::vector<u32> spirv; // as play binds it (after the set move and invariance)
    Shader::Info info;
    std::vector<GraphicsPipeline::CbufRead> cbuf_reads; // the checks a use must pass, as a variant's (StillHolds)
    std::vector<GraphicsPipeline::TypeRead> type_reads;
};

bool ShaderStoreOn();
bool ShaderStoreCheck();

/// A stage's key in the store: what its translation reads of the code memory (the instructions with the SPH, the
/// compiler constants - not the debug tag), the control fields, the stage, and `assumptions` (the caller's key
/// part for the runtime state it translated with).
std::string StoreKey(const std::vector<u8>& code, const ProgramStage& program, u32 stage, const std::string& assumptions);

/// Reads the index in the background (with the pipeline cache's load); a lookup waits for it.
void StartShaderStoreLoad();
std::optional<StoredStage> StoreGet(const std::string& key);
/// Adds an entry (the seed); one already there is kept. Written through to the file (flushed by StoreFlush).
void StorePut(const std::string& key, const StoredStage& entry);
void StoreFlush();
/// The tag the file must have; whether the file exists with it and holds entries (the seed's marker counts only then).
std::string StoreTag();
bool StoreFilled();

/// The bytes `info` is stored as (RT_SHADER_STORE_CHECK compares them).
std::vector<u8> InfoBytes(const Shader::Info& info);
/// The first of InfoFields' fields (by number, from 0) where `a` and `b` differ, for RT_SHADER_STORE_CHECK.
std::string InfoDifference(const Shader::Info& a, const Shader::Info& b);

} // namespace rt::nvn
