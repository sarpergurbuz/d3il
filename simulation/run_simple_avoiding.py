import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))  # repo root
# Also add D3IL environment package root so imports like `envs.*` resolve
sys.path.insert(0, str(repo_root / "environments" / "d3il"))
sys.path.insert(0, str(repo_root / "environments"))

# run_simple_avoiding.py
from avoiding_sim import Avoiding_Sim

class DummyAgent:
    def reset(self):
        return

if __name__ == "__main__":
    # Run multiple rollouts (including early-stops due to collision) and report successes
    N_ROLLOUTS = 10

    sim = Avoiding_Sim(seed=0, device="cpu", render=True, n_cores=1, n_trajectories=N_ROLLOUTS)
    agent = DummyAgent()

    successes, entropy = sim.test_agent(agent)

    # `successes` is a tensor of 0/1; sum to get number of successful runs
    try:
        num_success = int(successes.sum().item())
    except Exception:
        # fallback if successes is a numpy array
        num_success = int(successes.sum())

    print(f"Successful runs: {num_success}/{N_ROLLOUTS}")