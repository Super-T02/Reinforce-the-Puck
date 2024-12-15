# Project Structure

- **hockey_rl_project/**
    - **hockey_rl_project/**
        - `main__.py`
        - `__init__.py`
        - **components**
	        - **networks**
	        - **noise**
			    - gaussian noise
			    - ...
        - **agents/**
            - `__init__.py`
            - `agent_a.py`
            - `agent_b.py`
            - `base_agent.py`
        - **environments/**
            - `__init__.py`
            - `hockey_env_wrapper.py`
        - **training/**
            - `__init__.py`
            - `(general)trainer.py`
        - **evaluation/**
            - `__init__.py`
            - `evaluate_agents.py`
            - `plot_results.py`
        - **utils/**
            - `__init__.py`
            - `logger.py`
    - **tests/**
        - `__init__.py`
        - `test_agent_a.py`
        - `test_agent_b.py`
        - `test_hockey_env.py`
    - **configs/**
        - `agent_a_config.yaml`
        - `agent_b_config.yaml`
    - **experiments/**
        - **agent_a/**
        - **agent_b/**
    - **report/**
    - `.gitignore`
    - `pyproject.toml`
    - `README.md`
    - `requirements.txt`



# Serialisation
- Serialisation format for statistics: pickle
- Serialisation format for models: pth (see assignments)
