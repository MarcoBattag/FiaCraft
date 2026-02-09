from gym.envs.registration import register

# Definiamo l'ID univoco del tuo ambiente
register(
    id='FIA-WoodenPickaxe-v0',
    # Nota: togliamo il nome della classe se la funzione è definita fuori o gestita male
    entry_point='envs.FIAenv:fia_pickaxe_entrypoint', 
)