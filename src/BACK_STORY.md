# "Qwen3.6 Plus Preview" Rust port of gliner2

## Background

I wanted to port my Perfect Memory plugin for opencode to claw-code, and since claw-code is now in Rust (since 24h or so), I wanted to have rust-native GLiNER2 inference.

Fired up the Zed agent with OpenRouter's free offering of `qwen/qwen3.6-plus-preview:free` and got down to business.

## Prompt 1

    Mission: Port the `gliner2` inference to Rust. The goal is to be able to perform GLiNER2 inference in rust, training and other features are not important at this stage. Find the the GLiNER2 library in the GLiNER2 folder. Make a comprehensive plan, and store in PLAN.md.

Output: PLAN.md    

## Prompt 2

    If you feel confident that this plan can be implemented in its entirety, go ahead, work through it. Commit regularly, and keep the PLAN.md up-to-date with progress.
    
Output: 
  - 9999 lines of Rust via in 30 minutes

Context grew past 128K and the model started being very very slow. New context.


## Prompt 3

  The `PLAN.md` is almost up to date with the full architecture, you need to finish updating it since the previous session failed to proceed past section 4 of PLAN.md. The codebase has ~10,000 lines of un-tested foundational code ready for next steps.

  Let's start with updating the PLAN.md to be correct, and then try getting the project to build and finish it


Output:
  - Agent struggles for a bit with getting libpytorch / tch to play nice, eventually solves it
  - Agent proceeds to fix 56 cargo check errors - using a subagent!
    This is the perfect use of a subagent to not clutter main context; monotone iterative code-fixes with a clear goal.

```bash
cargo test

```
