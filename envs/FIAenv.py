from typing import List, Optional
import gym
from minerl.env import _singleagent
from minerl.herobraine import handlers
from minerl.herobraine.env_specs.basalt_specs import BasaltBaseEnvSpec, BasaltTimeoutWrapper, DoneOnESCWrapper

MINUTE = 20 * 60 

class FIAWoodenPickaxeEnvSpec(BasaltBaseEnvSpec):
    def __init__(self):
        super().__init__(
            name="FIA-WoodenPickaxe-v0",
            demo_server_experiment_name="wood_pickaxe",
            max_episode_steps=5 * MINUTE, # Aumentato a 5 min (è un task lungo per un agente)
            preferred_spawn_biome="forest",
            inventory=[], # Iniziamo vuoti
        )

    def create_mission_handlers(self):
        # Definiamo gli oggetti chiave per questo task
        target_items = ["log", "planks", "stick", "crafting_table", "wooden_pickaxe"]

        # Recuperiamo gli handler di base (movimento, camera, ecc.)
        base_handlers = super().create_mission_handlers()

        # Aggiungiamo gli handler specifici per il Piccone
        new_handlers = [
            # 1. Osservazione Inventario: Solo ciò che ci serve sapere
            handlers.FlatInventoryObservation(target_items),
            
            # 2. Crafting 2x2 (Inventario): Assi, Bastoncini, Tavolo
            handlers.CraftingAction(["planks", "stick", "crafting_table"]),
            
            # 3. Crafting 3x3 (Vicino al tavolo): CRUCIALE per il piccone!
            handlers.NearbyCraftAction(["wooden_pickaxe"]),
            
            # 4. Piazzare Blocchi: Necessario per mettere a terra il tavolo
            handlers.PlaceBlock(["crafting_table"]),
            
            # 5. Rompere Blocchi: Necessario per prendere il legno
            handlers.BreakBlock(["log"]),
            
            # 6. Ricompensa o Fine episodio quando otteniamo il piccone
            handlers.AgentQuitFromPossessingItem([dict(type="wooden_pickaxe", amount=1)])
        ]
        
        return base_handlers + new_handlers

    def create_observables(self) -> List[handlers.Handler]:
        # Definisce cosa vede la rete neurale
        return [
            handlers.POVObservation(self.resolution),
            handlers.FlatInventoryObservation(["log", "planks", "stick", "crafting_table", "wooden_pickaxe"])
        ]

    # Entrypoint per la registrazione
    @staticmethod
    def fia_pickaxe_entrypoint():
        # Nota: Importa qui la classe stessa se è in un file separato
        env_spec = FIAWoodenPickaxeEnvSpec()
        env = _singleagent._SingleAgentEnv(env_spec=env_spec)
        env = BasaltTimeoutWrapper(env)
        env = DoneOnESCWrapper(env)
        return env

# --- Esempio di registrazione (da mettere nel tuo main o __init__) ---
# gym.register(
#     id='FIA-WoodenPickaxe-v0',
#     entry_point='tuo_file:FIAWoodenPickaxeEnvSpec.fia_pickaxe_entrypoint',
# )