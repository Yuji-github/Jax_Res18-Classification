import jax
import jax.numpy as jnp
import flax
import numpy as np
import glob
import cv2
import optax
from tqdm import tqdm
import matplotlib.pyplot as plt


from cnn import CNN

cats = glob.glob('./0/*.jpg')
cats_y = np.zeros(len(cats))
cats_x = [cv2.imread(f)[..., ::-1] for f in cats]  # [...,::-1]  -> to read as RGB
cats_x = [cv2.resize(i/255, (256, 256)) for i in cats_x]  # normalizing and resizing

dogs = glob.glob('./1/*.jpg')
dogs_y = np.ones(len(dogs))
dogs_x = [cv2.imread(f)[..., ::-1] for f in dogs]  # [...,::-1]  -> to read as RGB
dogs_x = [cv2.resize(i/255, (256, 256)) for i in dogs_x]  # normalizing and resizing

train_x, train_y = np.concatenate((cats_x[:-2], dogs_x[:-2])), np.concatenate((cats_y[:-2], dogs_y[:-2]))
test_x, test_y = np.concatenate((cats_x[-2:], dogs_x[-2:])), np.concatenate((cats_y[-2:], dogs_y[-2:]))


# model
model = CNN()  # can be static (see in it from the training phase)

# initializing model
init_inputs = jnp.ones(shape=(1, 256, 256, 3))  # because jax is stateless (not like PyTorch [stateful])
init_rng = jax.random.PRNGKey(0)
variables = model.init(init_rng, init_inputs)
state, params = flax.core.pop(variables, 'params')
del variables

# optimizer
lr = 0.005
optimizer = optax.adam(lr)
opt_state = optimizer.init(params)

@jax.jit
def train_step(opt_state, batch, params, state, epoch):
    def _calculate_loss(params):
        x, y_true = batch
        y_pred, updated_state = model.apply({'params': params, **state}, x, show=True, epoch=epoch, mutable=list(state.keys()))
        loss = optax.sigmoid_binary_cross_entropy(jnp.float32([y_true]), y_pred).mean()
        return loss, updated_state

    (loss, updated_state), grads = jax.value_and_grad(_calculate_loss, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state)  # Defined below.
    params = optax.apply_updates(params, updates)
    return opt_state, params, updated_state, loss


losses = []
for i in tqdm(range(200)):
    opt_state, params, state, loss = train_step(opt_state, (jnp.array(train_x), jnp.array(train_y)), params, state, i)
    losses.append(loss)

plt.plot(losses)
plt.show()
