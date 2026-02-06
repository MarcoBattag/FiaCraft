from gym.envs.registration import register

# Definiamo l'ID univoco del tuo ambiente
ENV_ID = 'FIA-WoodenPickaxe-v0'

# Registriamo l'ambiente in Gym
register(
    id=ENV_ID,
    # Sintassi ENTRY_POINT: "cartella.nome_file:NomeClasse.nome_metodo_statico"
    # Nota: Assicurati che "fia_env.py" sia il nome reale del file nella cartella envs
    entry_point='envs.fia_env:FIAWoodenPickaxeEnvSpec.fia_pickaxe_entrypoint',
)