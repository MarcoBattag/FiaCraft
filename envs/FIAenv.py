from typing import List, Optional
import gym
from minerl.env import _singleagent
from minerl.herobraine.hero import handlers
from minerl.herobraine.env_specs.basalt_specs import BasaltBaseEnvSpec, BasaltTimeoutWrapper, DoneOnESCWrapper

MINUTE = 20 * 60 

# Lista completa delle varianti per non perdere nessun blocco raccolto
LOG_TYPES = ["log", "oak_log", "birch_log", "dark_oak_log", "spruce_log", "jungle_log", "acacia_log"]
PLANK_TYPES = ["planks", "oak_planks", "birch_planks", "dark_oak_planks", "spruce_planks", "jungle_planks", "acacia_planks"]
TARGET_ITEMS = LOG_TYPES + PLANK_TYPES + ["stick", "crafting_table", "wooden_pickaxe"]

class FIAWoodenPickaxeEnvSpec(BasaltBaseEnvSpec):
    def __init__(self):
        super().__init__(
            name="FIA-WoodenPickaxe-v0",
            demo_server_experiment_name="wood_pickaxe",
            max_episode_steps=5 * MINUTE,
            preferred_spawn_biome="forest",
            inventory=[], 
        )

    def create_mission_handlers(self):
        base_handlers = super().create_mission_handlers()

        new_handlers = [
            # 1. Fondamentale per sincronizzare l'inventario Java -> Python
            handlers.InventoryObservation(TARGET_ITEMS),
            handlers.ObserveItemStats(), 
            
            # 2. Crafting: permettiamo di creare planks da ogni tipo di legno
            # e oggetti base dalle planks
            handlers.CraftingAction(["planks", "stick", "crafting_table"]),
            
            # 3. Crafting avanzato per il piccone
            handlers.NearbyCraftAction(["wooden_pickaxe"]),
            
            # 4. Interazione con il mondo
            handlers.PlaceBlock(["crafting_table"]),
            
            # 5. Rompere blocchi: abilitiamo l'agente a rompere ogni tipo di albero
            handlers.BreakBlock(LOG_TYPES), 
            
            # 6. Condizione di vittoria
            handlers.AgentQuitFromPossessingItem([dict(type="wooden_pickaxe", amount=1)])
        ]
        
        return base_handlers + new_handlers

    def create_observables(self) -> List[handlers.Handler]:
        # Questa sezione definisce cosa viene passato alla rete neurale (il "Flat")
        return [
            handlers.POVObservation(self.resolution),
            # Usiamo TARGET_ITEMS per includere tutte le varianti di legno e assi
            handlers.FlatInventoryObservation(TARGET_ITEMS)
        ]

def fia_pickaxe_entrypoint():
    env_spec = FIAWoodenPickaxeEnvSpec()
    env = _singleagent._SingleAgentEnv(env_spec=env_spec)
    env = BasaltTimeoutWrapper(env)
    env = DoneOnESCWrapper(env)
    return env