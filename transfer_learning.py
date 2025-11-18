from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

FREEZE = True

def load_tl_model():
    return load_model("dqn_results/model.keras", compile=False)

class TL:
    def __init__(self, dqn_model):
        self.dqn_model = dqn_model
        self.freeze = FREEZE

    def build_tl_model(self):
        # freeze dei layer, tranne quello di output
        if self.freeze:
            for layer in self.dqn_model.layers[:-1]:
                layer.trainable = False

        alpha_lr = 1e-5
        self.dqn_model.compile(loss='mse', optimizer=Adam(learning_rate=alpha_lr))
        return self.dqn_model