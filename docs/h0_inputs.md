# h0 — Human inputs

<!-- Human-write only. The agent reads but does not edit this file.
     The ONLY place to paste raw input: prompts, chat snippets, constraints, direction.
     Group into sections; put dates in headings, e.g. "## 1. Kickoff prompt (2026-06-15)". -->

## 1. Kickoff prompt (2026-09-14)

This project exists because of a chain of events in the sibling `openvla` project
(`~/dev/openvla`, `docs/` there has the full history):

> "Can you put up the inference server for openvla box stacking? I was told to use GPU4 on
> gpu245 (aliased to the second machine now)" — led to discovering no box-stacking OpenVLA
> checkpoint exists yet; served the existing Ethernet-insertion checkpoint instead.

> "oh any changes I need to make to the yaml?"

> "all right, tried it; you can unhost the server now. The policy does move now, but it's too
> slow, which might be a model attribute? I was told to look into OPENVLA-OFT. Research to see
> if action-chunking can be done, if it cannot be done, we may drop this since we do have enough
> models in the paper now."

> "OK commit and push the docs, and we'll go forward with trying it"

> "OK, start with step 1."

> "Set up a docs structure in the new repo and I'll start a new agent from within it to handle
> the rest. Populate the docs with the needed background info"

The research referenced above lives in the sibling repo at
`~/dev/openvla/docs/04s_openvla_oft_feasibility.md` — read it for the full reasoning behind
every decision in this project's `04_plan.md`/`06_current.md`.

## 2. Task list (2026-09-14)

Same as the `openpi` project. General info as follows:

> Data path: `/cpfs01/data/devol/lerobot_vla_jepa/batch_20260909_203200_flexiv_action_superset_3cam`
> prompt: Stack the boxes  
> teleoperator: Jacky  
> station: flexiv 1  
> Arm: Dual  

> Data path: `/cpfs01/data/devol/lerobot_vla_jepa/batch_20260910_170153_flexiv_action_superset_3cam`
> prompt: Stack realsense boxes with only left arm
> teleoperator: `yeepeng`  
> station: flexiv 1  
> Arm: single arm  
> number of eps: 200  

> Data path: `/cpfs01/data/devol/lerobot_vla_jepa/batch_20260904_102647_flexiv_action_superset_3cam`
> prompt: mount the wire connector onto the DIN rail
> teleoperator: `nik`  
> station: flexiv 1  
> Arm: dual  
> number of eps: 400