#!/usr/bin/env python3

import sys
import pickle

import matplotlib.pyplot as plt
import numpy as np

from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop

from vaegan.models import create_models, build_graph
from vaegan.training import fit_models
from vaegan.data import shoes_loader, encoder_loader, decoder_loader, discriminator_loader, NUM_SAMPLES, mnist_loader
from vaegan.callbacks import DecoderSnapshot, ModelsCheckpoint


def set_trainable(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def main():
    encoder, decoder, discriminator = create_models()
    encoder_train, decoder_train, discriminator_train, vae, vaegan = build_graph(encoder, decoder, discriminator)
    
    encoder.load_weights('pretrained/encoder.030.h5')
    decoder.load_weights('pretrained/decoder.030.h5')
    discriminator.load_weights('pretrained/discriminator.030.h5')

    try:
        initial_epoch = int(sys.argv[1])
    except (IndexError, ValueError):
        initial_epoch = 0

    epoch_format = '.{epoch:03d}.h5'

    if initial_epoch != 0:
        suffix = epoch_format.format(epoch=initial_epoch)
        encoder.load_weights('encoder' + suffix)
        decoder.load_weights('decoder' + suffix)
        discriminator.load_weights('discriminator' + suffix)

    batch_size = 64
    rmsprop = RMSprop(lr=0.0005)

    set_trainable(encoder, False)
    set_trainable(decoder, False)
    discriminator_train.compile(rmsprop, ['binary_crossentropy'] * 3, ['acc'] * 3)
    discriminator_train.summary()

    set_trainable(discriminator, False)
    set_trainable(decoder, True)
    decoder_train.compile(rmsprop, ['binary_crossentropy'] * 2, ['acc'] * 2)
    decoder_train.summary()

    set_trainable(decoder, False)
    set_trainable(encoder, True)
    encoder_train.compile(rmsprop)
    encoder_train.summary()

    set_trainable(vaegan, True)

    checkpoint = ModelsCheckpoint(epoch_format, encoder, decoder, discriminator)
    decoder_sampler = DecoderSnapshot()

    callbacks = [checkpoint, decoder_sampler, TensorBoard()]

    epochs = 90

    steps_per_epoch = NUM_SAMPLES // batch_size

    seed = np.random.randint(2**32 - 1)

    img_loader = shoes_loader(batch_size, num_child=3, seed=seed)
    dis_loader = discriminator_loader(img_loader, seed=seed)
    dec_loader = decoder_loader(img_loader, seed=seed)
    enc_loader = encoder_loader(img_loader)

    models = [discriminator_train, decoder_train, encoder_train]
    generators = [dis_loader, dec_loader, enc_loader]
    metrics = [{'di_l': 1, 'di_l_t': 2, 'di_l_p': 3, 'di_a': 4, 'di_a_t': 7, 'di_a_p': 10}, {'de_l_t': 1, 'de_l_p': 2, 'de_a_t': 3, 'de_a_p': 5}, {'en_l': 0}]

    histories = fit_models(vaegan, models, generators, metrics, batch_size,
                           steps_per_epoch=steps_per_epoch, callbacks=callbacks,
                           epochs=epochs, initial_epoch=initial_epoch)

    with open('histories.pickle', 'wb') as f:
        pickle.dump(histories, f)

    x = next(shoes_loader(1))

    x_tilde = vae.predict(x)

    plt.subplot(211)
    plt.imshow((x[0].squeeze() + 1.) / 2.)

    plt.subplot(212)
    plt.imshow((x_tilde[0].squeeze() + 1.) / 2.)

    plt.show()


if __name__ == '__main__':
    main()
