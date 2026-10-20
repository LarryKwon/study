"""
Residual Connection과 Gradient 소실 실습
=========================================

문서 "Residual Connection과 Gradient 소실: 처음부터 정리"의 실습 코드입니다.
각 `# %%` 블록은 VS Code / Jupyter에서 셀 단위로 실행할 수 있고,
`python residual_gradient_lab.py`로 한 번에 돌려도 됩니다 (CPU 1~2분).

실습 구성
  Part 0. 역전파 기초 검증: ∂L/∂x = Jᵀ·∂L/∂y, nn.Linear 저장 모양
  Part 1. 층별 activation gradient norm: plain / residual / Post-LN / Pre-LN
  Part 2. identity 대 branch 분해 (상쇄가 일어나는지)
  Part 3. Jacobian 특이값 스펙트럼 (방향별 생존율)
  Part 4. null space와 방향별 소실 장난감 예제
  Part 5. Shattering: 가까운 입력끼리 gradient가 닮았나
  Part 6. 학습 후 weight gradient 비율과 lesion 실험

필요 패키지: torch, matplotlib
"""

# %% 공통 설정
import json
import os

import matplotlib

matplotlib.use("Agg")  # 화면 없이 PNG로 저장. 노트북에서는 이 줄을 지워도 됩니다.
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jacrev

torch.manual_seed(0)
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

D = 64        # residual stream 폭
DEPTH = 50    # 블록 수
BATCH = 128


# %% 모델 정의
class Block(nn.Module):
    """
    mode
      plain   : h ← relu(W h)                       (skip 없음, PyTorch 기본 초기화)
      plain_he: h ← relu(W h)                       (skip 없음, He 초기화로 norm은 유지되게)
      residual: h ← h + F(h)                        (LayerNorm 없음)
      post_ln : h ← LN(h + F(h))                    (원조 Transformer 방식)
      pre_ln  : h ← h + F(LN(h))                    (GPT-2 이후 표준)
    F(h) = W2 · relu(W1 · h)
    """

    def __init__(self, d, mode):
        super().__init__()
        self.mode = mode
        self.off = False  # Part 6 lesion: True면 branch를 끈다 (F = 0)
        if mode in ("plain", "plain_he"):
            self.lin = nn.Linear(d, d)
            if mode == "plain_he":
                nn.init.kaiming_normal_(self.lin.weight, nonlinearity="relu")
                nn.init.zeros_(self.lin.bias)
        else:
            self.f = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
            self.ln = nn.LayerNorm(d) if mode in ("post_ln", "pre_ln") else None

    def branch(self, h):
        """residual 계열에서 F 부분만 (Part 2 분해에 사용)"""
        if self.mode == "pre_ln":
            return self.f(self.ln(h))
        return self.f(h)

    def forward(self, h):
        if self.mode in ("plain", "plain_he"):
            return F.relu(self.lin(h))
        if self.off:  # lesion: h ← h (+ post_ln이면 LN만)
            return self.ln(h) if self.mode == "post_ln" else h
        if self.mode == "residual":
            return h + self.f(h)
        if self.mode == "post_ln":
            return self.ln(h + self.f(h))
        if self.mode == "pre_ln":
            return h + self.f(self.ln(h))
        raise ValueError(self.mode)


class Net(nn.Module):
    def __init__(self, d=D, depth=DEPTH, mode="residual"):
        super().__init__()
        self.mode = mode
        self.blocks = nn.ModuleList([Block(d, mode) for _ in range(depth)])
        self.head = nn.Linear(d, 1)
        self.final_ln = nn.LayerNorm(d) if mode == "pre_ln" else nn.Identity()

    def forward(self, x, log=None):
        """log가 dict이면 각 블록 출력 h_l에 hook을 걸어 ∂L/∂h_l을 기록"""
        h = x
        for i, blk in enumerate(self.blocks):
            h = blk(h)
            if log is not None and h.requires_grad:
                h.register_hook(lambda g, i=i: log.__setitem__(i, g.detach().clone()))
        return self.head(self.final_ln(h))


MODES = ["plain", "plain_he", "residual", "post_ln", "pre_ln"]


def make_data(n=BATCH, d=D, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    y = torch.randn(n, 1, generator=g)
    return x, y


# %% Part 0. 역전파 기초 검증
def part0():
    print("\n=== Part 0. 역전파 기초 검증 ===")

    # (a) 손으로 만든 함수: y1 = x1*x2, y2 = x1 + 3*x2
    x = torch.tensor([2.0, 5.0], requires_grad=True)
    f = lambda x: torch.stack([x[0] * x[1], x[0] + 3 * x[1]])
    J = jacrev(f)(x).detach()                 # 행 = 출력, 열 = 입력
    print("J =\n", J)                         # [[x2, x1], [1, 3]] = [[5, 2], [1, 3]]

    g = torch.tensor([1.0, 0.1])              # ∂L/∂y 를 (a, b) = (1, 0.1)로 가정
    y = f(x)
    (grad_x,) = torch.autograd.grad(y, x, grad_outputs=g)
    print("autograd ∂L/∂x  :", grad_x)
    print("Jᵀ · g (세로)    :", J.T @ g)
    print("g · J  (가로)    :", g @ J)        # 같은 숫자
    print("J · g  (틀린 배치):", J @ g)       # 행과 짝지어져서 다른 숫자

    # (b) nn.Linear의 저장 모양과 역전파 식
    layer = nn.Linear(4, 3)
    print("\nnn.Linear(4, 3).weight.shape =", tuple(layer.weight.shape), " ← (out, in)")
    X = torch.randn(5, 4, requires_grad=True)
    Y = layer(X)
    G = torch.randn_like(Y)                   # 임의의 upstream gradient
    Y.backward(G)
    W = layer.weight.detach()
    print("grad_x == G @ W        :", torch.allclose(X.grad, G @ W, atol=1e-6))
    print("grad_W == G.T @ X      :", torch.allclose(layer.weight.grad, G.T @ X.detach(), atol=1e-6))
    print("Linear의 Jacobian == W :", torch.allclose(jacrev(lambda v: layer(v))(X[0].detach()), W, atol=1e-6))


# %% Part 1. 층별 activation gradient norm
def grad_norms(model, x, y):
    log = {}
    model.zero_grad()
    loss = F.mse_loss(model(x, log=log), y)
    loss.backward()
    # 샘플당 RMS: 원소 수에 영향받지 않게 norm / sqrt(원소 수)
    return [log[i].pow(2).mean().sqrt().item() for i in range(len(model.blocks))]


def part1():
    print("\n=== Part 1. 층별 activation gradient (RMS) ===")
    x, y = make_data()
    results = {}
    for mode in MODES:
        torch.manual_seed(0)
        model = Net(mode=mode)
        norms = grad_norms(model, x, y)
        results[mode] = norms
        ratio = norms[0] / norms[-1]
        print(f"{mode:9s}  첫 블록 {norms[0]:.2e} | 마지막 블록 {norms[-1]:.2e} | 첫/마지막 = {ratio:.2e}")

    plt.figure(figsize=(7, 4.2))
    for mode, norms in results.items():
        plt.semilogy(range(1, DEPTH + 1), norms, label=mode)
    plt.xlabel("block index l (1 = input side)")
    plt.ylabel("RMS of dL/dh_l (log)")
    plt.title("Activation gradient by depth at init")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/part1_grad_norm_by_depth.png", dpi=150)
    plt.close()
    return results


# %% Part 2. identity 대 branch 분해
def part2(mode="residual"):
    """∂L/∂h_l = g_{l+1} (identity) + J_Fᵀ g_{l+1} (branch) 를 블록마다 분리"""
    print(f"\n=== Part 2. identity 대 branch 분해 ({mode}) ===")
    torch.manual_seed(0)
    model = Net(mode=mode)
    x, y = make_data()

    # 순전파하면서 각 블록의 입력 h_{l-1}을 저장, 출력 쪽 gradient는 hook으로
    inputs, log = [], {}
    h = x
    for i, blk in enumerate(model.blocks):
        inputs.append(h.detach())
        h = blk(h)
        h.register_hook(lambda g, i=i: log.__setitem__(i, g.detach()))
    F.mse_loss(model.head(model.final_ln(h)), y).backward()

    rows = []
    for i, blk in enumerate(model.blocks):
        h_in = inputs[i].clone().requires_grad_()
        g_out = log[i]                                     # ∂L/∂h_{l+1}
        (g_branch,) = torch.autograd.grad(blk.branch(h_in), h_in, grad_outputs=g_out)
        ratio = (g_branch.norm() / g_out.norm()).item()
        cos = F.cosine_similarity(g_branch.flatten(), g_out.flatten(), dim=0).item()
        total = ((g_out + g_branch).norm() / g_out.norm()).item()
        rows.append({"block": i + 1, "branch/identity": ratio, "cos": cos, "total_gain": total})

    for r in rows[:: max(1, DEPTH // 10)]:
        print(f"block {r['block']:2d}: ‖branch‖/‖identity‖ = {r['branch/identity']:.3f}, "
              f"cos = {r['cos']:+.3f}, 이 블록 통과 배율 = {r['total_gain']:.3f}")
    prod = 1.0
    for r in rows:
        prod *= r["total_gain"]
    print(f"블록 통과 배율의 곱 (end-to-end 근사) = {prod:.3f}")
    print("해석: cos ≈ -1 & 비율 ≈ 1 → 상쇄(소실), 비율 ≪ 1 → branch가 거의 기여 안 함")
    return rows


# %% Part 3. Jacobian 특이값 스펙트럼
def end_to_end_jacobian(model, x0):
    """h_L = blocks(h_0) 의 Jacobian ∂h_L/∂h_0 (샘플 하나, D×D)"""

    def f(h):
        for blk in model.blocks:
            h = blk(h)
        return h

    return jacrev(f)(x0)


def part3():
    print("\n=== Part 3. end-to-end Jacobian 특이값 ===")
    x, _ = make_data(n=1)
    spectra = {}
    plt.figure(figsize=(7, 4.2))
    for mode in MODES:
        torch.manual_seed(0)
        model = Net(mode=mode)
        J = end_to_end_jacobian(model, x[0]).detach()
        s = torch.linalg.svdvals(J)
        spectra[mode] = s.tolist()
        print(f"{mode:9s}  최대 {s.max():.2e} | 중앙값 {s.median():.2e} | 최소 {s.min():.2e} | "
              f"0.01 미만 방향 {int((s < 0.01).sum())}/{len(s)}")
        plt.semilogy(range(1, len(s) + 1), s.clamp_min(1e-30), label=mode)  # 정확히 0인 값만 1e-30으로 표시
    plt.axhline(1.0, color="gray", ls="--", lw=1)
    plt.xlabel("singular value rank")
    plt.ylabel("singular value (log)")
    plt.title("Spectrum of end-to-end Jacobian dh_L/dh_0")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/part3_jacobian_spectrum.png", dpi=150)
    plt.close()

    # 큰 모델에서 쓰는 근사: 랜덤 벡터 v에 대해 ‖Jᵀv‖/‖v‖ (전체 Jacobian 없이 VJP만)
    torch.manual_seed(0)
    model = Net(mode="pre_ln")
    h0 = x[0].clone().requires_grad_()
    h = h0
    for blk in model.blocks:
        h = blk(h)
    gains = []
    for _ in range(32):
        v = torch.randn_like(h)
        (u,) = torch.autograd.grad(h, h0, grad_outputs=v, retain_graph=True)
        gains.append((u.norm() / v.norm()).item())
    t = torch.tensor(gains)
    print(f"[VJP 근사, pre_ln] ‖Jᵀv‖/‖v‖ 평균 {t.mean():.3f}, 표준편차 {t.std():.3f}")
    print("해석: plain_he는 Part 1의 norm은 멀쩡한데 64개 방향 중 대부분이 죽어 있다.\n"
          "      post_ln의 죽은 방향 2개는 LayerNorm의 '평균 방향'(1,1,…,1)과 '크기 방향'(출력 자신)이다.\n"
          "      LN 출력은 항상 평균 0, 크기 일정이라 그 두 방향으로는 움직일 수 없고,\n"
          "      역전파에서도 그 두 방향 성분의 gradient는 0이 된다.")
    return spectra


# %% Part 4. null space와 방향별 소실 장난감 예제
def part4():
    print("\n=== Part 4. null space와 방향별 소실 ===")
    # (a) y = x1 + x2 : (1, -1) 방향은 출력에 영향이 없음 → gradient에도 그 성분이 없음
    W = torch.tensor([[1.0, 1.0]])
    for g in [torch.tensor([0.7]), torch.tensor([-2.0])]:
        grad_x = W.T @ g
        along_null = grad_x @ torch.tensor([1.0, -1.0]) / 2**0.5
        print(f"g = {g.item():+.1f} → ∂L/∂x = {grad_x.tolist()}, (1,-1) 방향 성분 = {along_null.item():.1f}")
    print("svdvals([1, 1]) =", torch.linalg.svdvals(W).tolist(), " (≈ √2, 나머지 방향은 null space)")

    # (b) diag(1, 0.1)을 10번 통과: norm은 그대로인데 두 번째 방향은 1e-10배
    A = torch.diag(torch.tensor([1.0, 0.1]))
    g = torch.tensor([1.0, 1.0])
    for _ in range(10):
        g = A.T @ g
    print(f"diag(1, 0.1)ᵀ 를 10번 곱한 gradient = {g.tolist()}")
    print(f"norm은 {torch.tensor([1.,1.]).norm():.3f} → {g.norm():.3f} (덜 줄어 보임), "
          f"두 번째 방향은 1e-10배로 소실")


# %% Part 5. Shattering
def part5():
    """
    입력 공간의 한 직선 위 점들에서 ∂output/∂x를 구하고,
    이웃한 점끼리 gradient가 얼마나 닮았는지(cosine) 본다.
    Balduzzi et al. (2017): skip 없는 깊은 망은 이웃끼리도 gradient가 거의 무관(백색잡음)해진다.
    """
    print("\n=== Part 5. Shattering (이웃 입력 간 gradient 유사도) ===")
    n = 256
    a, b = torch.randn(D), torch.randn(D)
    t = torch.linspace(-1, 1, n).unsqueeze(1)
    xs = a + t * b                                   # 직선 위의 n개 점
    out = {}
    for mode in MODES:
        torch.manual_seed(0)
        model = Net(mode=mode)
        x = xs.clone().requires_grad_()
        model(x).sum().backward()
        g = x.grad.double()
        g = g / g.norm(dim=1, keepdim=True).clamp_min(1e-300)  # 크기가 1e-19여도 방향만 비교
        cos_nb = (g[:-1] * g[1:]).sum(1).mean().item()
        cos_far = (g[: n // 2] * g[n // 2 :]).sum(1).mean().item()
        out[mode] = {"neighbor_cos": cos_nb, "far_cos": cos_far}
        print(f"{mode:9s}  이웃 간 cos = {cos_nb:.3f} | 멀리 떨어진 점 간 cos = {cos_far:.3f}")
    print("해석: 이웃 간 cos가 1에서 멀어질수록 gradient가 노이즈처럼 '산산조각' 난 상태.\n"
          "      plain은 gradient 크기가 1e-19 수준으로 이미 소실돼서 방향 비교가 의미 없음.\n"
          "      plain_he는 norm은 유지되는데(Part 1) 방향이 산산조각 난다 → norm만 보면 놓치는 경우")
    return out


# %% Part 6. 학습 후 weight gradient 비율과 lesion
def part6(mode="pre_ln", steps=300):
    print(f"\n=== Part 6. 학습 후 weight gradient 비율과 lesion ({mode}) ===")
    torch.manual_seed(0)
    # 학습할 만한 목표: 랜덤 teacher 네트워크의 출력
    teacher = nn.Sequential(nn.Linear(D, 128), nn.Tanh(), nn.Linear(128, 1))
    X = torch.randn(2048, D)
    with torch.no_grad():
        Y = teacher(X)

    model = Net(mode=mode)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(steps):
        idx = torch.randint(0, len(X), (BATCH,))
        loss = F.mse_loss(model(X[idx]), Y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        base = F.mse_loss(model(X), Y).item()
    print(f"{steps} step 학습 후 전체 loss = {base:.4f}")

    # (a) weight gradient 상대 크기 ‖∇W‖/‖W‖ (블록 첫 Linear 기준)
    opt.zero_grad()
    F.mse_loss(model(X[:BATCH]), Y[:BATCH]).backward()
    ratios = [(b.f[0].weight.grad.norm() / b.f[0].weight.norm()).item() for b in model.blocks]
    print("‖∇W‖/‖W‖  앞 5개 블록:", [f"{r:.1e}" for r in ratios[:5]])
    print("‖∇W‖/‖W‖  뒤 5개 블록:", [f"{r:.1e}" for r in ratios[-5:]])

    # (b) lesion: 블록 하나의 branch를 끄고(F=0) loss가 얼마나 오르나
    increases = []
    for blk in model.blocks:
        blk.off = True
        with torch.no_grad():
            l = F.mse_loss(model(X), Y).item()
        blk.off = False
        increases.append(l - base)
    worst = sorted(range(DEPTH), key=lambda i: -increases[i])[:5]
    print("블록 하나를 껐을 때 loss 증가, 가장 큰 5개:",
          [(i + 1, round(increases[i], 4)) for i in worst])
    worst_loss = base + max(increases)
    print(f"평균 증가 {sum(increases) / DEPTH:.4f} (학습된 loss {base:.4f}, "
          f"평균만 예측하는 '망가진' 수준 {Y.var().item():.4f})")
    print(f"가장 중요한 블록을 꺼도 loss {worst_loss:.4f} → 망가진 수준의 {worst_loss / Y.var().item():.1%}")
    print("해석: 하나 꺼도 망가진 수준에 한참 못 미치면, 깊은 한 경로가 아니라 "
          "짧은 경로들의 앙상블처럼 동작 (Veit et al., 2016)")

    plt.figure(figsize=(7, 3.6))
    plt.bar(range(1, DEPTH + 1), increases)
    plt.xlabel("lesioned block")
    plt.ylabel("loss increase")
    plt.title(f"Lesion study ({mode}, trained {steps} steps)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/part6_lesion.png", dpi=150)
    plt.close()
    return {"base_loss": base, "grad_ratio": ratios, "lesion_increase": increases}


# %% 전체 실행
if __name__ == "__main__":
    part0()
    r1 = part1()
    r2 = part2("residual")
    r2b = part2("post_ln")
    r3 = part3()
    part4()
    r5 = part5()
    r6 = part6()
    with open(f"{OUT}/results.json", "w") as fp:
        json.dump({"part1": r1, "part2_residual": r2, "part2_post_ln": r2b,
                   "part3": r3, "part5": r5, "part6": r6}, fp, ensure_ascii=False, indent=1)
    print(f"\n그림과 숫자는 ./{OUT}/ 에 저장되었습니다.")
