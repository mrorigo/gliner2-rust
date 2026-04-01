# "Qwen3.6 Plus Preview" LLM one-shot Rust port of gliner2

## Background

I wanted to port my Perfect Memory plugin for opencode to claw-code, and since claw-code is now in Rust (since 24h or so), I wanted to have rust-native GLiNER2 inference.

Fired up the Zed agent with OpenRouter's free offering of `qwen/qwen3.6-plus-preview:free` and got down to business.

## Prompt 1

    Mission: Port the `gliner2` inference to Rust. The goal is to be able to perform GLiNER2 inference in rust, training and other features are not important at this stage. Find the the GLiNER2 library in the GLiNER2 folder. Make a comprehensive plan, and store in PLAN.md.

Output: PLAN.md    

## Prompt 2

    If you feel confident that this plan can be implemented in its entirety, go ahead, work through it. Commit regularly, and keep the PLAN.md up-to-date with progress.
    
Output: 7500 lines of Rust


```bash
cargo test

```
