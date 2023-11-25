from dataclasses import dataclass
import torch




####### SETTING UP THE CONFIGURATION FILE #######

@dataclass
class Config():

    VECTOR_SIZE: int = 300
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"





if __name__ == "__main__":
    print("Confguration file for TransformerGNN ")
    config = Config()
    print(config)
    



