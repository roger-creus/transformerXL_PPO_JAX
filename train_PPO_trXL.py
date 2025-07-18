import time
import os
import jax
import jax.numpy as jnp
import wandb
import matplotlib.pyplot as plt
from trainer_PPO_trXL import make_train

# ✅ Define base defaults
base_config = {
    "OPTIMIZER": "adam",
    "LR": 2e-4,
    "NUM_ENVS": 1024,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 1e5,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 8,
    "GAMMA": 0.999,
    "GAE_LAMBDA": 0.8,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.002,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 1.0,
    "ACTIVATION": "relu",
    "ENV_NAME": "craftax",
    "ANNEAL_LR": True,
    "qkv_features": 256,
    "EMBED_SIZE": 256,
    "num_heads": 8,
    "num_layers": 2,
    "hidden_layers": 256,
    "WINDOW_MEM": 128,
    "WINDOW_GRAD": 64,
    "gating": True,
    "gating_bias": 2.0,
    "seed": 0,
}

def main():
    run = wandb.init(project="test", config=base_config)
    config = dict(wandb.config)
    seed = config["seed"]
    optimizer = config["OPTIMIZER"]
    prefix = f"results_craftax_{optimizer}_{run.name}"
    

    try:
        os.makedirs(prefix, exist_ok=True)
    except Exception as e:
        print(f"Directory creation failed for {prefix}: {e}")

    print("Start compiling and training")
    time_a = time.time()
    rng = jax.random.PRNGKey(seed)
    train_jit = jax.jit(make_train(config))
    print("Compiled")
    out = train_jit(rng)
    duration = time.time() - time_a
    print("training and compilation took", duration)

    # ✅ Plot episode returns
    returns = out["metrics"]["returned_episode_returns"]
    plt.plot(returns)
    plt.xlabel("Updates")
    plt.ylabel("Return")
    plot_path = f"{prefix}/return_{seed}.png"
    plt.savefig(plot_path)
    plt.clf()

    # ✅ Save model and training data
    jnp.save(f"{prefix}/{seed}_params", out["runner_state"][0].params)
    jnp.save(f"{prefix}/{seed}_config", config)
    jnp.save(f"{prefix}/{seed}_metrics", out["metrics"])

    # ✅ Log to wandb (mean of last 25 values)
    mean_return = float(jnp.mean(returns[-25:]))
    run.log({
        "mean_return": mean_return,
    })
    run.save(plot_path)
    run.save(f"{prefix}/{seed}_*.npy")
    run.finish()

if __name__ == "__main__":
    main()
