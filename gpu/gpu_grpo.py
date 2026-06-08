"""Real LoRA-GRPO trainer for the RLVR reward-partition study.

Trains ONE (model, reward_type) configuration with genuine GRPO on GSM8K and
logs held-out pass@1. Designed to be launched one-per-GPU in parallel (see
gpu_grpo_launch.sh). This is the real-RL counterpart to the tabular simulator
and the best-of-N proxy: it reproduces the elicitation-vs-reward-design
partition under actual policy-gradient training.

Reward types (matching src/grpo.py in the simulator):
  frozen   : no update; reports the base model's pass@1 (a_F)
  true     : +1 iff the extracted answer equals the gold answer        (a_T)
  random   : Bernoulli(0.5) independent of correctness                 (a_R)
  spurious : +1 iff the answer equals the sampled-group MAJORITY answer (a_S)
             (self-consistency / TTRL-style pseudo-reward)

GRPO update: group-relative advantages A = (r - mean_g)/(std_g + eps), a
token-level policy-gradient loss over completion tokens, and a k3 KL penalty to
the frozen reference (LoRA disabled) for stability. Dynamic-sampling filtering
drops degenerate (all-equal-reward) groups.

Outputs: <out>/grpo_<model>_<reward>.json  with the eval curve and final acc.
"""
import os, sys, json, re, time, argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

ANS_RE = re.compile(r"answer is\D*(-?[\d,]+)", re.IGNORECASE)


def extract_pred(text):
    m = ANS_RE.search(text)
    if m:
        s = m.group(1)
    else:
        nums = re.findall(r"-?\d[\d,]*", text)
        if not nums:
            return None
        s = nums[-1]
    try:
        return int(s.replace(",", ""))
    except ValueError:
        return None


def gold(ans):
    return int(ans.split("####")[-1].strip().replace(",", ""))


def build_prompt(tok, q):
    msgs = [{"role": "user", "content": q + "\nThink step by step and end with 'The answer is <number>.'"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def evaluate(model, tok, eval_items, n_eval, max_new, dev, eval_samples=1):
    """pass@1 over n_eval held-out problems (greedy)."""
    model.eval()
    correct = 0
    for q, a in eval_items[:n_eval]:
        enc = tok(build_prompt(tok, q), return_tensors="pt").to(dev)
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        txt = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        correct += int(extract_pred(txt) == a)
    model.train()
    return correct / n_eval


def compute_rewards(reward_type, preds, gold_ans, rng):
    if reward_type == "true":
        return np.array([1.0 if p == gold_ans else 0.0 for p in preds])
    if reward_type == "random":
        return rng.integers(0, 2, size=len(preds)).astype(float)
    if reward_type == "spurious":
        valid = [p for p in preds if p is not None]
        if not valid:
            return np.zeros(len(preds))
        vals, counts = np.unique(valid, return_counts=True)
        maj = int(vals[np.argmax(counts)])
        return np.array([1.0 if p == maj else 0.0 for p in preds])
    raise ValueError(reward_type)


def main(a):
    dev = "cuda"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    short = a.model.split("/")[-1]
    tag = f"{short}_{a.reward}"

    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev)
    if a.reward != "frozen":
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.train()

    ds_tr = load_dataset("openai/gsm8k", "main", split="train")
    ds_te = load_dataset("openai/gsm8k", "main", split="test")
    train_items = [(ds_tr[i]["question"], gold(ds_tr[i]["answer"])) for i in range(len(ds_tr))]
    eval_idx = np.random.default_rng(123).choice(len(ds_te), size=a.n_eval, replace=False)
    eval_items = [(ds_te[int(i)]["question"], gold(ds_te[int(i)]["answer"])) for i in eval_idx]

    base_acc = evaluate(model, tok, eval_items, a.n_eval, a.max_new, dev)
    curve = [{"step": 0, "acc": base_acc}]
    print(f"[{tag}] base pass@1 = {base_acc:.3f}", flush=True)

    if a.reward == "frozen":
        json.dump({"model": a.model, "reward": a.reward, "base_acc": base_acc,
                   "final_acc": base_acc, "curve": curve}, open(f"{a.out}/grpo_{tag}.json", "w"), indent=2)
        print(f"[{tag}] frozen done", flush=True)
        return

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        batch = [train_items[int(i)] for i in rng.integers(0, len(train_items), size=a.B)]
        prompts = [build_prompt(tok, q) for q, _ in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
        plen = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=a.max_new, do_sample=True,
                                 temperature=a.temp, top_p=0.95, num_return_sequences=a.G,
                                 pad_token_id=tok.pad_token_id)
        # gen: (B*G, plen+L). group i = rows [i*G:(i+1)*G]
        comp = gen[:, plen:]
        texts = tok.batch_decode(comp, skip_special_tokens=True)
        adv_all = torch.zeros(gen.shape[0], device=dev)
        keep = torch.zeros(gen.shape[0], dtype=torch.bool, device=dev)
        for i, (q, ga) in enumerate(batch):
            sl = slice(i * a.G, (i + 1) * a.G)
            preds = [extract_pred(t) for t in texts[sl]]
            r = compute_rewards(a.reward, preds, ga, rng)
            if r.std() < 1e-6:   # dynamic-sampling filter: drop degenerate groups
                continue
            adv = (r - r.mean()) / (r.std() + 1e-6)
            adv_all[sl] = torch.tensor(adv, device=dev, dtype=torch.float32)
            keep[sl] = True
        if keep.sum() == 0:
            continue
        attn = (gen != tok.pad_token_id).long()
        cmask = attn[:, plen:].float()
        tgt = gen[:, plen:]                       # completion target tokens
        keep_idx = torch.nonzero(keep, as_tuple=True)[0]
        total_tok = (cmask[keep_idx]).sum().clamp(min=1.0)

        def token_logprobs(ids, am, with_grad):
            """Per-completion-token logprob, memory-safe via fused cross-entropy."""
            ctx = torch.enable_grad() if with_grad else torch.no_grad()
            with ctx:
                logits = model(input_ids=ids, attention_mask=am).logits[:, plen - 1:-1]
                V = logits.shape[-1]
                nll = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, V).float(), ids[:, plen:].reshape(-1),
                    reduction="none").view(ids.shape[0], -1)
            return -nll  # logprob of taken tokens

        # reference logprobs (no grad, adapters disabled), micro-batched
        ref_lp = torch.zeros_like(cmask)
        mb = a.micro_bs
        with torch.no_grad(), model.disable_adapter():
            for s in range(0, gen.shape[0], mb):
                ref_lp[s:s+mb] = token_logprobs(gen[s:s+mb], attn[s:s+mb], False)

        # policy forward + loss, micro-batched with gradient accumulation over KEPT rows
        opt.zero_grad()
        loss_val = 0.0
        for s in range(0, len(keep_idx), mb):
            rows = keep_idx[s:s+mb]
            lp = token_logprobs(gen[rows], attn[rows], True)
            kl = torch.exp(ref_lp[rows] - lp) - (ref_lp[rows] - lp) - 1.0
            pg = -adv_all[rows].unsqueeze(1) * lp
            per_tok = (pg + a.beta * kl) * cmask[rows]
            loss_mb = per_tok.sum() / total_tok
            loss_mb.backward()
            loss_val += loss_mb.item()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); opt.zero_grad()

        if step % a.eval_every == 0 or step == a.steps:
            acc = evaluate(model, tok, eval_items, a.n_eval, a.max_new, dev)
            curve.append({"step": step, "acc": acc})
            print(f"[{tag}] step {step}/{a.steps} kept={int(keep.sum())}/{gen.shape[0]} "
                  f"loss={loss_val:.3f} pass@1={acc:.3f} ({(time.time()-t0)/step:.1f}s/step)", flush=True)

    final = curve[-1]["acc"]
    json.dump({"model": a.model, "reward": a.reward, "base_acc": base_acc,
               "final_acc": final, "curve": curve,
               "config": {"steps": a.steps, "B": a.B, "G": a.G, "lr": a.lr,
                          "beta": a.beta, "temp": a.temp, "max_new": a.max_new, "n_eval": a.n_eval}},
              open(f"{a.out}/grpo_{tag}.json", "w"), indent=2)
    print(f"[{tag}] DONE base={base_acc:.3f} final={final:.3f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--reward", required=True, choices=["frozen", "true", "random", "spurious"])
    p.add_argument("--out", default="results_grpo")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--B", type=int, default=6)
    p.add_argument("--G", type=int, default=6)
    p.add_argument("--micro_bs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.02)
    p.add_argument("--temp", type=float, default=0.9)
    p.add_argument("--max_new", type=int, default=200)
    p.add_argument("--n_eval", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
