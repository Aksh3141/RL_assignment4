configs = {

    'InvertedPendulum-v4': {

        "PPO": {
            "hyperparameters": {
                "algorithm": "PPO",
                "total_timesteps": 500_000,
                "log_interval": 200,
                "eval_freq": 5000,
                "save_freq": 20000,
                "Run_name": "PPO_InvPend",

                # PPO best hyperparams
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "policy_kwargs": {
                    "net_arch": [
                        dict(pi=[64, 64], vf=[64, 64])
                    ]
                },
            },
        },

        "A2C": {
            "hyperparameters": {
                "algorithm": "A2C",
                "total_timesteps": 300_000,
                "log_interval": 200,
                "eval_freq": 5000,
                "save_freq": 20000,
                "Run_name": "A2C_InvPend",

                # A2C best hyperparams
                "learning_rate": 7e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "n_steps": 5,
                "n_envs": 16,
                "policy_kwargs": {
                    "net_arch": [
                        dict(pi=[64, 64], vf=[64, 64])
                    ]
                },
            },
        },
    },


    'HalfCheetah-v4': {

        "PPO": {
            "hyperparameters": {
                "algorithm": "PPO",
                "total_timesteps": 1_500_000,
                "log_interval": 200,
                "eval_freq": 10000,
                "save_freq": 50000,
                "Run_name": "PPO_HC",

                # BEST PPO hyperparams for HalfCheetah
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 256,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,

                "policy_kwargs": {
                    "net_arch": [
                        dict(pi=[256, 256, 256], vf=[256, 256, 256])
                    ]
                },
            },
        },

        "A2C": {
            "hyperparameters": {
                "algorithm": "A2C",
                "total_timesteps": 1_000_000,
                "log_interval": 200,
                "eval_freq": 10000,
                "save_freq": 50000,
                "Run_name": "A2C_HC",

                # A2C tuned hyperparams
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "n_steps": 20,
                "n_envs": 16,
                "policy_kwargs": {
                    "net_arch": [
                        dict(pi=[128, 128], vf=[128, 128])
                    ]
                },
            },
        },
    },


    'Hopper-v4': {

        "PPO": {
            "hyperparameters": {
                "algorithm": "PPO",
                "total_timesteps": 1_000_000,
                "log_interval": 200,
                "eval_freq": 10000,
                "save_freq": 50000,
                "Run_name": "PPO_Hopper",

                # PPO tuned hyperparams
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 20,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "policy_kwargs": {
                    "net_arch": [
                        dict(pi=[128, 128], vf=[128, 128])
                    ]
                },
            },
        },

        "A2C": {
            "hyperparameters": {
                "algorithm": "A2C",
                "total_timesteps": 800_000,
                "log_interval": 200,
                "eval_freq": 10000,
                "save_freq": 50000,
                "Run_name": "A2C_Hopper",

                # A2C tuned hyperparams
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "n_steps": 20,
                "n_envs": 16,
                "policy_kwargs": {
                    "net_arch": [
                        dict(pi=[128, 128], vf=[128, 128])
                    ]
                },
            },
        },
    },
}
