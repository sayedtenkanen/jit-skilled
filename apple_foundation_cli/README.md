# jitskilled-apple-fm

A tiny Swift CLI that bridges `jitskilled`'s Python pipeline to Apple's
on-device `FoundationModels` framework (Apple Intelligence).

This only works on a real Mac -- it cannot be built or run in a Linux
sandbox, which is why it's a separate Swift package rather than something
this repo's CI can verify. Treat it as a reference implementation to build
and adjust on your machine.

## Requirements

- macOS 26 (Tahoe) or newer, with Apple Intelligence turned on and the
  on-device model downloaded (Settings > Apple Intelligence & Siri).
- Xcode 26+ / a matching Swift toolchain.

## Build

```bash
cd apple_foundation_cli
swift build -c release
```

The binary is written to `.build/release/jitskilled-apple-fm`.

## Wire it up to jitskilled

Either put the binary on your `PATH` as `jitskilled-apple-fm`, or point the
Python side at it directly:

```bash
export APPLE_FM_CLI_PATH="$(pwd)/apple_foundation_cli/.build/release/jitskilled-apple-fm"
python -m jitskilled run --llm apple --mode skill \
  --slot_library configs/slots_v1.yaml --run_name apple_fm_run
```

## Manual smoke test

```bash
echo '{"system": "Reply with one word.", "user": "Say hello."}' \
  | .build/release/jitskilled-apple-fm
# -> {"text":"Hello"}
```

## Contract

stdin: one JSON object `{"system": str, "user": str, "max_tokens": int?}`.
stdout on success: one JSON object `{"text": str}`.
On failure: a message on stderr and a non-zero exit code.

Keep this contract stable if you modify `main.swift` -- it's what
`jitskilled/llm_apple.py` depends on.

## If this doesn't compile

Apple's `FoundationModels` API (`LanguageModelSession`, `GenerationOptions`,
`SystemLanguageModel`) was introduced at WWDC 2025 and may have evolved
since this was written. Check Apple's current FoundationModels
documentation for the current method signatures and adjust `main.swift` --
the JSON stdin/stdout contract above is the only part
`jitskilled/llm_apple.py` actually depends on.
