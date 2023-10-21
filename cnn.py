from flax import linen as nn
#import matplotlib.pyplot as plt


class CNN(nn.Module):
    @nn.compact  # to skip setup()
    def __call__(self, inputs, train=False, show=False, epoch=0):
        x = nn.Conv(features=3, kernel_size=(7, 7), strides=1)(inputs)  # kernel_size requires tuple
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        add_x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=3, kernel_size=(3, 3), strides=1)(add_x)  # kernel_size requires tuple
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = x + add_x

        # if show and epoch%20==0:
        #     plt.imshow(x.primal[0])
        #     plt.show()

        x = x.reshape((x.shape[0], -1))  # flatten for dense layers
        x = nn.Dense(features=128)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(features=1)(x)
        return nn.sigmoid(x)
