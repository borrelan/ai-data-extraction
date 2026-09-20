# AI Coding Assistant Corpus and Training-Data Toolkit

Extract local Codex, Claude, Gemini, Cursor, Continue, OpenCode, Prime Agent,
Pi/Oh My Pi, Trae, and Windsurf histories, then normalize them through one
privacy-gated boundary for SFT, explicit preference, trajectory, and
prompt-only RL preparation. Raw exports are evidence and provenance; they are
not trainer input.

## 🎯 What This Does

The extractor layer discovers conversation history including:
- ✅ User messages & AI responses
- ✅ Code context (file paths, line numbers, snippets)
- ✅ Code diffs and suggested edits
- ✅ Multi-file contexts
- ✅ Tool use and execution results
- ✅ Timestamps and metadata

The canonical builder then:

- removes reasoning fields and tagged reasoning blocks;
- normalizes roles, tool calls, tool results, diffs, context, and optional tool schemas;
- adds deterministic IDs, splits, provider/task/outcome/privacy tags, and provenance hashes;
- preflights quality per source session before chunking, then stamps every
  segment with the same provider-neutral gate and `session_quality_id`;
- preserves only explicit rewards and chosen/rejected pairs; it never invents labels;
- writes `sft.jsonl`, `trajectories.jsonl`, `tool_traces.jsonl`,
  `action_windows.jsonl`, `preferences.jsonl`, `rl_prompts.jsonl`,
  `rejected.jsonl`, and a manifest.

See [`docs/TRAINING_DATA.md`](docs/TRAINING_DATA.md) for the contract and
[`docs/TOOL_USE_TRAINING.md`](docs/TOOL_USE_TRAINING.md) for the tool-use,
DevOps, skill/MCP, and RL-facing data contract.
The harness-owned skill/MCP/permission trace is specified in
[`docs/HARNESS_TRACE_CONTRACT.md`](docs/HARNESS_TRACE_CONTRACT.md).
The proposed long-session event/episode/action-window contract is in
[`docs/SCHEMA_DESIGN.md`](docs/SCHEMA_DESIGN.md).

Prime Agent and Pi-family sources are retained in explicit provenance lanes:
ordinary sessions are `optional_alt`, while advisor overlays are
`quarantine`. This is not a blanket quality judgment. Each inspected session
also receives a provider-neutral quality gate and flags; local-model
provenance is recorded as evidence, not treated as failure. The default
builder includes only `primary`; opt into another lane with
`--training-lanes optional_alt` after the session-level quality review.

Task-specific research decisions, audits, source inventories, run checkpoints,
and evaluation notes are kept in the local gitignored `.tmp/` directory and
are not product documentation or trainer input.

## 📦 Included Scripts

### 1. `extract_claude_code.py`
Extracts from Claude Code / Claude Desktop
- **Searches**: `~/.claude`, `~/.claude-code`, `~/.claude-local`, `~/.claude-m2`, `~/.claude-zai`
- **Formats**: JSONL session files
- **Includes**: Messages, tool use, file contexts, diffs

### 2. `extract_codex.py`
Extracts from Codex (if installed)
- **Searches**: `~/.codex`, `~/.codex-local`
- **Formats**: Rollout JSONL files
- **Includes**: User/agent messages, tool results, diffs

### 3. `extract_cursor.py`
Extracts from Cursor (Chat + Composer + Agent) - ALL VERSIONS
- **Searches**: `~/Library/Application Support/Cursor` (macOS) or equivalent
- **Formats**: SQLite databases (`state.vscdb`, `cursorDiskKV`)
- **Handles**:
  - Old Chat mode (workspace storage)
  - Composer inline storage (v1.x - messages in composerData array)
  - Composer separate storage (v1.x-v2.0 transition - messages in bubbleId keys)
  - Latest Composer/Agent (v2.0+)
- **Includes**:
  - Code context, selections, diffs
  - Suggested edits and code blocks
  - Tool results and execution outputs

### 4. `extract_trae.py`
Extracts from Trae
- **Searches**: `~/.trae`, `~/Library/Application Support/Trae`
- **Formats**: JSONL and SQLite databases
- **Includes**: Chat, agent data, tool use, diffs

### 5. `extract_windsurf.py`
Extracts from Windsurf
- **Searches**: `~/Library/Application Support/Windsurf` or equivalent
- **Formats**: SQLite databases (VSCode-like format)
- **Includes**: Chat, agent/flow conversations, code context

### 6. `extract_continue.py`
Extracts from Continue AI Assistant
- **Searches**: `~/.continue/sessions/`
- **Formats**: JSON session files
- **Includes**:
  - User/assistant messages
  - Tool calls and results
  - Raw reasoning blocks (removed by the canonical builder)
  - Context items
  - Workspace information

### 7. `extract_gemini.py`
Extracts from Google Gemini CLI
- **Searches**: `~/.gemini/tmp/[hash]/chats/`
- **Formats**: JSON session files
- **Includes**:
  - User/assistant messages
  - Raw thoughts (removed by the canonical builder)
  - Token usage breakdown
  - Model information
  - Project hash and workspace linking

### 8. `extract_opencode.py`
Extracts from OpenCode (CLI + Desktop)
- **Searches**: 
  - CLI: `~/.local/share/opencode/` (Linux), `~/Library/Application Support/opencode` (macOS)
  - Desktop: `~/.local/share/ai.opencode.app` (Linux), `~/Library/Application Support/ai.opencode.app` (macOS)
- **Formats**: SQLite (`opencode.db`), JSON files (sessions/messages/parts), and Tauri .dat files (desktop)
- **Includes**:
  - User/assistant messages with full conversation hierarchy
  - Tool calls and tool results
  - Code blocks and text content
  - Token usage and cost tracking
  - Model and provider information
  - Agent mode and session metadata
  - Project directory and version info
  - Parent/child session relationships
- **Handles**:
  - newer OpenCode CLI installs that store conversations in `opencode.db`
  - legacy JSON storage layouts under `storage/message`, `storage/part`, and `storage/session`
  - sidecar metadata from `storage/session_diff`, `storage/directory-readme`, `storage/agent-usage-reminder`, and `storage/rules-injector`

### 9. `extract_agent_sessions.py`
Extracts Prime Agent, Pi, and Oh My Pi JSONL session streams.
- **Searches**: `~/.prime/agent/sessions`, Prime subagent session artifacts,
  `~/.omp/agent/sessions`, discovered Oh My Pi backups, and narrow standalone
  Pi session locations
- **Includes**: visible user/assistant/tool messages, tool-call IDs,
  bounded arguments/observations, source hashes, and chunk lineage
- **Omits**: thinking blocks and raw harness/advisor control payloads
- **Lanes**: Prime/ordinary Pi sessions are `optional_alt`; `__advisor.jsonl`
  is `quarantine` and is excluded from the default build
- **Quality**: every session gets `candidate`, `review_required`, or
  `quarantine` from structural evidence independent of provider/model; use
  `--quality-overrides` for an explicit per-session review decision

### 10. `extract_cursor_cli.py`
Extracts from the `cursor-agent` terminal CLI (the Composer/Agent history that the GUI
extractor in `extract_cursor.py` does not cover)
- **Searches**: `~/.cursor/chats/<chatId>/<agentId>/store.db` (all platforms)
- **Formats**: content-addressed SQLite blob store
  - `blobs` table: `id` (sha256 hex) → `data`; message blobs are JSON (`{role, content, id, ...}`),
    tree nodes are protobuf-framed lists of 32-byte child blob-id refs
  - `meta` table: hex-encoded JSON with `agentId`, `latestRootBlobId`, `name`, `mode`, `createdAt`
- **Includes**:
  - Ordered messages reconstructed by DFS-walking refs from `latestRootBlobId`
  - System/user/assistant/tool turns with full content
  - Chat/agent ids, title, mode, created-at, code-context and diff flags

## 🚀 Quick Start

### Installation

```bash
# Extractors and the canonical builder use the Python 3 standard library.
python3 --version  # Python 3.10+ is required
```

### Basic Usage

```bash
# Extract from Claude Code
python3 extract_claude_code.py

# Extract from Cursor
python3 extract_cursor.py

# Extract from Codex
python3 extract_codex.py

# Include Codex .jsonl.backup stores as an explicitly tagged source class
INCLUDE_CODEX_BACKUPS=1 python3 extract_codex.py

# Extract from Trae
python3 extract_trae.py

# Extract from Windsurf
python3 extract_windsurf.py

# Extract from Continue
python3 extract_continue.py

# Extract from Gemini CLI
python3 extract_gemini.py

# Extract from OpenCode
python3 extract_opencode.py

# Extract Prime Agent / Pi / Oh My Pi (advisor overlays are quarantined)
python3 extract_agent_sessions.py

# Run snapshot-backed adapters without changing process HOME or reading live stores
python3 extract_claude_code.py \
  --source-home /path/to/snapshot/home \
  --source-manifest /path/to/source_manifest.json \
  --output-dir extracted_data
python3 extract_gemini.py --source-home /path/to/snapshot/home \
  --source-manifest /path/to/source_manifest.json \
  --output-dir extracted_data
python3 extract_agent_sessions.py --source-home /path/to/snapshot/home \
  --source-manifest /path/to/source_manifest.json \
  --output-dir extracted_data
python3 extract_opencode.py --source-home /path/to/snapshot/home \
  --source-manifest /path/to/source_manifest.json \
  --output-dir extracted_data
python3 extract_codex.py --source-home /path/to/snapshot/home \
  --source-manifest /path/to/source_manifest.json \
  --output extracted_data/codex.jsonl

# Apply explicit per-session quality decisions without changing source lanes
python3 extract_agent_sessions.py \
  --quality-overrides .tmp/quality_overrides.json

# Extract from Cursor CLI (cursor-agent)
python3 extract_cursor_cli.py

# Extract from ALL tools at once
./extract_all.sh

# extract_all.sh forwards SOURCE_HOME and SOURCE_MANIFEST to adapters that
# support stable-snapshot admission; set INCLUDE_CODEX_BACKUPS=1 explicitly
# when backup sessions are in scope.
SOURCE_HOME=/path/to/snapshot/home \
SOURCE_MANIFEST=/path/to/source_manifest.json \
INCLUDE_CODEX_BACKUPS=1 \
./extract_all.sh

# Inventory every discovered store without copying source content
python3 inventory_sources.py --output source_inventory.json

# Preflight JSONL and backup lines before the expensive canonical build
python3 preflight_sources.py extracted_data --output .tmp/source_preflight.json \
  --overwrite

# Build review-gated datasets from raw exports without external dependencies
python3 build_training_data.py extracted_data --output-dir training_data

# Apply provider-neutral per-session quality decisions at the canonical boundary
python3 build_training_data.py extracted_data --output-dir training_data_reviewed \
  --quality-overrides .tmp/quality_overrides.json \
  --quality-gates candidate,review_required

# Build a separately reviewed optional-alt lane (never mix by accident)
python3 build_training_data.py extracted_data \
  --output-dir training_data_optional_alt \
  --training-lanes optional_alt \
  --quality-gates candidate

# Include sessions awaiting manual quality review in an explicitly named audit build
python3 build_training_data.py extracted_data \
  --output-dir training_data_optional_alt_review \
  --training-lanes optional_alt \
  --quality-gates candidate,review_required

# Optional: model-assisted privacy filtering, then bind the exact manifest
python3 -m pip install -r requirements-privacy-filter.txt
python3 filter_privacy.py extracted_data --output-dir filtered_data --device cuda
python3 build_training_data.py filtered_data \
  --output-dir training_data \
  --privacy-mode filtered \
  --privacy-manifest filtered_data/privacy_manifest.json \
  --privacy-approved
```

### Output

Extractors create timestamped raw JSONL files under `extracted_data/`. The
builder writes separate, ignored training outputs under `training_data/`:

```
extracted_data/
├── claude_code_conversations_20250116_143022.jsonl
├── cursor_complete_20250116_143045.jsonl
├── gemini_conversations_20250116_143145.jsonl
├── codex_conversations_20250116_143102.jsonl
├── trae_conversations_20250116_143115.jsonl
├── windsurf_conversations_20250116_143130.jsonl
├── continue_conversations_20250116_143145.jsonl
├── opencode_conversations_20250116_143200.jsonl
├── prime_agent_sessions_20250116_143205.jsonl
├── pi_sessions_20250116_143205.jsonl
└── pi_advisor_quarantine_20250116_143205.jsonl
```

The `training_data/` directory is deliberately not a raw merge: its files have
different contracts and must be consumed by dataset type.

## 📊 Output Format

Extractor output is provider-specific JSONL. It may contain reasoning,
identifiers, paths, secrets, tool payloads, and incomplete turns; keep it local.
Long Codex sessions are emitted as multiple bounded provider records with
source-event lineage instead of one giant JSONL line. The canonical builder is
still the only trainer-data mapper.
The canonical `sft.jsonl` record is instead shaped like:

```json
{
  "schema_version": "ai-data-extraction/v1",
  "example_id": "sha256:...",
  "dataset": "sft",
  "split": "train",
  "messages": [
    {
      "role": "user",
      "content": "How do I fix this TypeScript error?"
    },
    {
      "role": "assistant",
      "content": "Use a string value or change the declared type."
    }
  ],
  "tags": ["provider:cursor", "task:debugging", "privacy:review"],
  "quality": {"status": "review"},
  "privacy": {"eligible_for_training": false}
}
```

## 🔍 How It Works

### Auto-Discovery Process

Each script follows this pattern:

1. **Detect Operating System** (macOS, Linux, Windows)
2. **Search Common Locations**:
   - macOS: `~/Library/Application Support`, `~/.config`, `~/`
   - Linux: `~/.config`, `~/.local/share`, `~/`
   - Windows: `%APPDATA%`, `%LOCALAPPDATA%`, `~/`
3. **Find All Installations** of the target tool
4. **Scan Storage Locations**:
   - SQLite databases (`.vscdb`, `.db`)
   - JSONL session files
   - Project-specific directories
5. **Extract Complete Data** including context and diffs
6. **Save to Organized JSONL** with timestamps

### Storage Formats Handled

#### Claude Code / Codex
- **Format**: JSONL files (one event per line)
- **Location**: `~/.claude/projects/[project]/[session].jsonl`
- **Structure**: Event-based with type markers

#### Cursor (v0.43 - v2.0+)
- **Format**: SQLite databases
- **Locations**:
  - Workspace: `~/Library/Application Support/Cursor/User/workspaceStorage/[hash]/state.vscdb`
  - Global: `~/Library/Application Support/Cursor/User/globalStorage/state.vscdb`
- **Tables**: `ItemTable` (Chat), `cursorDiskKV` (Composer/Agent)
- **Storage Evolution**:
  - **v0.x - v1.x**: Chat mode in workspace `ItemTable`
  - **v1.x**: Composer inline (messages in `composerData.conversation[]`)
  - **v1.x - v2.0 transition**: Composer separate (messages in `bubbleId:{composer}:{bubble}` keys)
  - **v2.0+**: Latest format with enhanced metadata
- **Keys**:
  - `workbench.panel.aichat.view.aichat.chatdata` (Chat mode)
  - `composerData:{uuid}` (Composer metadata + conversation)
  - `bubbleId:{composer}:{bubble}` (Individual messages - transitional format)
  - `codeBlockDiff:{id}` (Code block diffs)

#### Trae / Windsurf
- **Format**: Hybrid (JSONL + SQLite)
- **Location**: Similar to VSCode/Cursor structure
- **Structure**: VSCode extension data format

## 🎓 Understanding the Data

### Message Roles
- `user`: Human developer messages
- `assistant`: AI assistant responses

### Code Context Fields
- `code_context`: File selections and code snippets
- `suggested_diffs`: AI-proposed code changes
- `tool_use`: Code execution, file operations
- `tool_results`: Execution outputs, diffs applied
- `diff_histories`: Full edit history

### Metadata Fields
- `source`: Which tool (e.g., "cursor-composer", "claude-code")
- `session_id`/`composer_id`: Unique conversation ID
- `project_path`: Working directory
- `timestamp`: Message time
- `model`: AI model used (if available)

## 🔧 Advanced Usage

### Normalize all extractions

```bash
# Raw files are not concatenated. Normalize and deduplicate them instead.
python3 build_training_data.py extracted_data --output-dir training_data \
  --oversize-strategy chunk

# Inspect counts, rejection reasons, hashes, and eligibility
python3 -m json.tool training_data/manifest.json
```

Long sessions are parent containers, not single examples. The default builder
splits them into bounded, linked segments and emits `action_windows.jsonl` for
tool-transition review; `--oversize-strategy reject` is available only for
comparison with the old whole-record policy. See
[`docs/TRAINING_DATA.md`](docs/TRAINING_DATA.md) for the trainer contract.

### Create Agent Skills from a corpus

`corpus_to_skills.py` samples canonical or filtered conversations and asks an
OpenAI-compatible chat-completions model to synthesize reusable
[Agent Skills](https://agentskills.io/specification). Each result is written as
`<skill-name>/SKILL.md` with validated `name` and `description` metadata.

The command is local-first: unless `OPENAI_BASE_URL` is set, it connects to
`http://127.0.0.1:8000/v1`. Point it at a locally served model and a filtered
corpus:

```bash
python3 corpus_to_skills.py filtered_data \
  --model your-local-model \
  --skills 5 \
  --output-dir generated_skills
```

You can also set `SKILLS_MODEL` instead of passing `--model`. The script uses
only Python's standard library, samples across the corpus with a configurable
character budget, treats corpus content as untrusted data, validates the model's
JSON response, and refuses to overwrite existing skills unless `--overwrite`
is passed.

Review every generated skill before installing it. Filter the extracted corpus
first, especially before explicitly configuring a remote API endpoint; model
output can still repeat sensitive source material despite prompt-level guards.

### Filter by Date

```python
import json
from datetime import datetime

with open('extracted_data/cursor_complete_20250116.jsonl') as f:
    for line in f:
        conv = json.loads(line)
        created = conv.get('created_at', 0)
        if created > 1704067200000:  # After Jan 1, 2024
            print(json.dumps(conv))
```

### Extract Only Conversations with Diffs

```python
import json

with open('extracted_data/cursor_complete.jsonl') as f:
    for line in f:
        conv = json.loads(line)
        if any('suggested_diffs' in m or 'diff_histories' in m
               for m in conv['messages']):
            print(json.dumps(conv))
```

## 📋 Data Quality

### What Gets Extracted

✅ **Complete Conversations**:
- Both user prompts AND AI responses
- Multi-turn dialogues
- Full conversation context

✅ **Code Context**:
- File paths and names
- Selected code snippets
- Line number ranges
- Multi-file selections

✅ **Diffs and Edits**:
- Suggested code changes
- Applied diffs
- Edit histories
- File modifications

✅ **Metadata**:
- Timestamps
- Project paths
- Model information
- Conversation names

### What Might Be Missing

⚠️ **Partial Data**:
- Conversations without AI responses (user-only)
- Deleted or archived sessions
- Corrupted database entries

⚠️ **Privacy Considerations**:
- May include proprietary code
- May include API keys/secrets
- May include personal file paths

## 🛡️ Privacy & Security

### Filter a corpus with OpenAI Privacy Filter

The optional `filter_privacy.py` command runs the official
[`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) model
locally and replaces detected spans with typed placeholders such as
`<PRIVATE_EMAIL>` and `<SECRET>`. It recursively filters every string value, so
messages, code, tool output, diffs, and path metadata are all covered while the
JSON structure stays intact.

The privacy filter requires Python 3.10+ and downloads the model weights on its
first run (about 2.8 GB). Install the pinned official OpenAI runtime separately:

```bash
python3 -m pip install -r requirements-privacy-filter.txt
```

Filter individual JSONL files or a whole extraction directory:

```bash
python3 filter_privacy.py extracted_data --output-dir filtered_data

# Use CUDA when available
python3 filter_privacy.py extracted_data --output-dir filtered_data --device cuda
```

Inputs are never modified. Existing outputs are refused unless `--overwrite`
is supplied. The unfiltered text stays local: the runtime downloads the model
from Hugging Face and performs inference on the selected CPU or CUDA device.

Privacy Filter is a data-minimization aid, not an anonymization or compliance
guarantee. Review filtered output and combine it with secret scanning and your
organization's privacy controls before sharing or training.

### Before Using Extracted Data

1. **Scan for Secrets**:
```bash
pip install detect-secrets
detect-secrets scan extracted_data/*.jsonl
```

2. **Review Sensitive Data**:
- Check for API keys, passwords, tokens
- Verify no proprietary code exposed
- Sanitize file paths if needed

3. **Storage**:
- Keep on encrypted drives
- Don't commit to public repositories
- Secure backups recommended

## 🎯 Training Use Cases

### Direct Fine-Tuning

```python
from datasets import load_dataset

dataset = load_dataset(
    'json',
    data_files='training_data/sft.jsonl',
    split='train'
)

# The builder already rejects incomplete conversations. Keep only records that
# passed the explicit privacy gate.
dataset = dataset.filter(lambda x: x['privacy']['eligible_for_training'])
```

Use `training_data/preferences.jsonl` for DPO-style training and
`training_data/rl_prompts.jsonl` only as prompt input for an environment-backed
RL run. The trajectory file is an offline audit/label surface, not a reward
function.

### Same-State Preference Training

`build_agent_preference_curriculum.py` composes trainer-neutral
`prompt`/`chosen`/`rejected` rows only when both alternatives share the exact
observable prompt state. It excludes hidden reasoning, malformed alternatives,
evaluation-prompt overlap, duplicate pair identities, and parent leakage across
train and validation splits. The output is immutable: the builder refuses an
existing destination and binds every JSONL file in `manifest.json`.

```bash
python3 build_agent_preference_curriculum.py \
  --skill-release /path/to/qualified-skill-sft-release \
  --sft-release /path/to/qualified-balanced-sft-release \
  --when2call-source /path/to/pinned/when2call_train_pref.jsonl \
  --when2call-revision <immutable-revision> \
  --evaluation-cases /path/to/held-out-cases.jsonl \
  --blocked-text-pattern '<forbidden-model-facing-pattern>' \
  --output-dir /path/to/new-preference-release
```

For long-horizon agent tuning, `build_agent_preference_curriculum_v2.py`
shifts the mix from generic tool/no-tool choices to exact-state recovery,
follow-through, skill routing, permission, premature-stop, and verified-finish
preferences. It requires the exact target tokenizer while selecting rows,
caps each Open-SWE parent to one row per split and behavior family, and refuses
a release if generic replay is not the minority or one source exceeds half of
either split. Source-executed Open-SWE transitions remain explicitly distinct
from locally replayed verifier evidence.

Run the builder in the same pinned container used by the DPO runtime so the
selection-time tokenizer and independent preflight tokenizer are identical:

```bash
docker run --rm --network=none \
  --user "$(id -u):$(id -g)" \
  --workdir /workspace --env HOME=/tmp --env HF_HOME=/tmp/hf \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env PYTHONPATH=/workspace \
  --mount type=bind,src="$PWD",dst=/workspace,readonly \
  --mount type=bind,src=/data-120,dst=/data-120 \
  --entrypoint python ai-data-extraction/unsloth-xpu-dpo:trl028 \
  /workspace/build_agent_preference_curriculum_v2.py \
  --skill-release /path/to/qualified-skill-sft-release \
  --sft-release /path/to/qualified-balanced-sft-release \
  --when2call-source /path/to/pinned/when2call_train_pref.jsonl \
  --when2call-revision <immutable-revision> \
  --evaluation-cases /path/to/held-out-cases.jsonl \
  --model-dir /data-120/models/Qwen3.5-9B \
  --blocked-text-pattern '<forbidden-model-facing-pattern>' \
  --output-dir /path/to/new-v2-preference-release
```

The local Qwen3.5-9B Intel XPU reference runtime is under `runtime/dpo/`.
Its image pins TRL by wheel hash, validates rows with the exact tokenizer and
chat template, pretokenizes both branches without truncation, loads the same
SFT adapter as trainable policy and frozen reference, and refuses success unless
the policy changed, the reference did not, and the saved PEFT adapter reloads.
`launch_local.sh` is intentionally bound to the qualified local paths, image
digest, B70 device, and baseline service; update those constants as one reviewed
runtime contract rather than overriding individual stages ad hoc.

```bash
docker build -t ai-data-extraction/unsloth-xpu-dpo:trl028 runtime/dpo
runtime/dpo/launch_local.sh --preflight-only
runtime/dpo/launch_local.sh
```

Preference accuracy is only a trainer health signal. Promotion still requires
a held-out, matched-seed behavior comparison and executable long-horizon task
verification; do not infer agent improvement from training loss or preference
margin alone.

### With Unsloth

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/qwen2.5-coder-7b-instruct",
    max_seq_length=4096,
    load_in_4bit=True,
)

def format_chat(example):
    return {
        'text': tokenizer.apply_chat_template(
            example['messages'],
            tokenize=False
        )
    }

dataset = dataset.map(format_chat)
```

## 🐛 Troubleshooting

### No installations found

**Problem**: Script reports "No installations found"

**Solutions**:
1. Check if the tool is actually installed
2. Verify installation location manually
3. Add custom path to script:
```python
# Add to find_XXX_installations() function
locations.append(Path("/custom/path/to/tool"))
```

### Empty extracted_data directory

**Problem**: Extraction completes but no data found

**Solutions**:
1. Verify you've actually used the tool and have chat history
2. Check if data is in a non-standard location
3. Look for database files manually:
```bash
find ~ -name "*.vscdb" -o -name "*.db" 2>/dev/null
```

### Database locked errors

**Problem**: SQLite database is locked

**Solutions**:
1. Close the AI tool before running extraction
2. Use read-only mode:
```python
conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
```

### Permission denied

**Problem**: Cannot read certain files

**Solutions**:
1. Run with appropriate permissions
2. Check file ownership
3. Copy databases to accessible location first

## 📚 Platform-Specific Notes

### macOS
- Uses `~/Library/Application Support` for most tools
- May need Full Disk Access for system directories
- SQLite databases typically in `~/Library/Application Support/[Tool]/User/`

### Linux
- Uses `~/.config` and `~/.local/share`
- Check `~/.local/state` for some tools
- May use `$XDG_CONFIG_HOME` if set

### Windows
- Uses `%APPDATA%` and `%LOCALAPPDATA%`
- Paths: `C:\Users\[User]\AppData\Roaming\[Tool]`
- May need admin privileges for Program Files

## 🔄 Version Compatibility

### Cursor
- ✅ v2 (0.43+): Composer/Agent in `cursorDiskKV`
- ✅ v1: Chat in workspace `ItemTable`
- ⚠️ Pre-v0.43: Different format, limited support

### Claude Code
- ✅ All versions with JSONL session files
- ✅ Project-based structure

### Codex
- ✅ Rollout JSONL format
- ✅ Time-based session organization

## 📈 Performance Tips

### Large Datasets
```bash
# Process a canonical file in chunks only after the manifest has been recorded
split -l 1000 training_data/sft.jsonl training_sft_chunk_

# Compress for storage
gzip extracted_data/*.jsonl
```

### Speed Optimization
```python
# Use multiprocessing for large scans
from multiprocessing import Pool

with Pool() as pool:
    results = pool.map(extract_from_db, db_files)
```

## 🤝 Contributing

Found a new storage format or tool? Contributions welcome!

1. Follow existing script structure
2. Add auto-discovery logic
3. Extract complete data (messages + context + diffs)
4. Output to organized JSONL
5. Update this README

## 📄 License

MIT License - Use freely for training ML models

## ⚠️ Disclaimer

This toolkit extracts YOUR OWN data from locally installed AI tools. Users are responsible for:
- Ensuring they have rights to extracted data
- Handling sensitive/proprietary information appropriately
- Complying with tool Terms of Service
- Scanning for secrets before sharing/training

---

**Updated**: September 16, 2026
**Status**: Audited implementation; privacy review, source coverage, and model evaluation remain release gates
**Compatibility**: Python 3.10+, macOS/Linux/Windows
