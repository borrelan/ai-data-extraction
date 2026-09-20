#!/usr/bin/env bash
# Extract provider-specific raw records, then send them through the one
# canonical training-data builder. Raw records remain private and are never
# concatenated into a trainer input file.

set -uo pipefail

raw_output_dir="${RAW_OUTPUT_DIR:-extracted_data}"
training_output_dir="${TRAINING_OUTPUT_DIR:-training_data}"
mkdir -p "$raw_output_dir"

declare -a found_tools=()
declare -a not_found=()
declare -a failed_tools=()

run_extractor() {
    local label="$1"
    local script="$2"
    local log_name="$3"
    shift 3
    local -a extra_args=("$@")
    local log_path="$raw_output_dir/$log_name"
    local status

    echo "🔍 Extracting from $label..."
    set +e
    EXTRACTED_DATA_DIR="$raw_output_dir" python3 "$script" "${extra_args[@]}" 2>&1 | tee "$log_path"
    status=${PIPESTATUS[0]}

    if [[ "$status" -eq 0 ]] && grep -Eq 'Total conversations([^0-9]|$)*[1-9]|Total conversations extracted([^0-9]|$)*[1-9]|Found [1-9]|Extracted [1-9]' "$log_path"; then
        found_tools+=("$label")
    else
        not_found+=("$label")
    fi
    if [[ "$status" -ne 0 ]]; then
        failed_tools+=("$label (exit $status)")
    fi
    echo
}

declare -a source_args=()
if [[ -n "${SOURCE_HOME:-}" ]]; then
    source_args+=(--source-home "$SOURCE_HOME")
fi
if [[ -n "${SOURCE_MANIFEST:-}" ]]; then
    source_args+=(--source-manifest "$SOURCE_MANIFEST")
fi

run_extractor "Claude Code" extract_claude_code.py claude_extraction.log "${source_args[@]}"
run_extractor "Cursor" extract_cursor.py cursor_extraction.log
run_extractor "Cursor CLI" extract_cursor_cli.py cursor_cli_extraction.log
run_extractor "Codex" extract_codex.py codex_extraction.log "${source_args[@]}"
run_extractor "Trae" extract_trae.py trae_extraction.log
run_extractor "Windsurf" extract_windsurf.py windsurf_extraction.log
run_extractor "Continue" extract_continue.py continue_extraction.log
run_extractor "Gemini CLI" extract_gemini.py gemini_extraction.log "${source_args[@]}"
run_extractor "OpenCode" extract_opencode.py opencode_extraction.log "${source_args[@]}"
run_extractor "Prime Agent / Pi / Oh My Pi" extract_agent_sessions.py agent_sessions_extraction.log "${source_args[@]}"

echo "================================================================================"
echo "EXTRACTION SUMMARY"
echo "================================================================================"
if ((${#found_tools[@]} > 0)); then
    echo "✅ Extractors with data-like output:"
    for tool in "${found_tools[@]}"; do
        echo "   - $tool"
    done
fi
if ((${#not_found[@]} > 0)); then
    echo "⚠️  Extractors with no confirmed data or an error:"
    for tool in "${not_found[@]}"; do
        echo "   - $tool"
    done
fi

if ((${#failed_tools[@]} > 0)); then
    echo
    echo "❌ One or more extractors failed; refusing to build from a partial run:"
    for tool in "${failed_tools[@]}"; do
        echo "   - $tool"
    done
    exit 1
fi

echo
echo "🔐 Building canonical training datasets from raw JSONL..."
builder_args=("$raw_output_dir" "--output-dir" "$training_output_dir")
builder_args+=("--training-lanes" "${TRAINING_LANES:-primary}")
builder_args+=("--quality-gates" "${QUALITY_GATES:-candidate}")
builder_args+=("--model-tiers" "${MODEL_TIERS:-tier1_frontier}")
if [[ "${OVERWRITE_TRAINING_DATA:-0}" == "1" ]]; then
    builder_args+=("--overwrite")
fi

if python3 build_training_data.py "${builder_args[@]}"; then
    echo "✅ Canonical datasets written to $training_output_dir"
else
    echo "❌ Canonical dataset build failed; inspect extractor logs and rejected.jsonl." >&2
    exit 1
fi

echo
echo "Raw JSONL files are kept under $raw_output_dir for provenance."
echo "Use filter_privacy.py before setting --privacy-approved for training eligibility."
