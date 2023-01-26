import tensorflow as tf
import matplotlib.pyplot as plt
from functools import partial

# uncomment this if you want to show debug information
# tf.debugging.set_log_device_placement(True)

DATASET_FROM_KAGGLE_PATH = "../PozeKaggle/AugmentedAlzheimerDataset"
DATASET_FROM_GEORGE = "../AugmentedDataset"
DEFAULT_DATASET_PATH = DATASET_FROM_GEORGE

EPOCHS = 10
VGG16_INCEPTION_V3_RESNET50_EPOCHS = 5

BATCH_SIZE = 64
VGG16_MODEL_BATCH_SIZE = 8
INCEPTION_V3_MODEL_BATCH_SIZE = 8
RESNET50_MODEL_BATCH_SIZE = 8

IMG_HEIGHT = 210
IMG_WIDTH = 210
NUM_CLASSES = 4
SEED = 123
RGB_DEPTH = 3


def change_batch_size(new_batch_size):
    global BATCH_SIZE
    BATCH_SIZE = new_batch_size


def change_epoch(new_epoch):
    global EPOCHS
    EPOCHS = new_epoch


def init_train_dataset():
    return tf.keras.utils.image_dataset_from_directory(
        DEFAULT_DATASET_PATH + "/Train",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )


def init_validation_dataset():
    return tf.keras.utils.image_dataset_from_directory(
        DEFAULT_DATASET_PATH + "/Validation",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )


def init_test_dataset():
    return tf.keras.utils.image_dataset_from_directory(
        DEFAULT_DATASET_PATH + "/Test",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )


def plot_some_images_from_train_ds():
    TRAIN_DATASET = init_train_dataset()
    VALIDATION_DATASET = init_validation_dataset()
    TEST_DATASET = init_test_dataset()

    plt.figure(figsize=(10, 10))
    class_names = TRAIN_DATASET.class_names
    for images, labels in TRAIN_DATASET.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()


# 4 simple models with different activation functions
def simple_model_activation_relu():
    return tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, RGB_DEPTH)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CLASSES)
    ])


def simple_model_activation_leaky_relu():
    return tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(32, (3, 3),
                               activation=partial(tf.nn.leaky_relu, alpha=0.01),
                               input_shape=(IMG_HEIGHT, IMG_WIDTH, RGB_DEPTH)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CLASSES)
    ])


def simple_model_activation_sigmoid():
    return tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(32, (3, 3),
                               activation='sigmoid',
                               input_shape=(IMG_HEIGHT, IMG_WIDTH, RGB_DEPTH)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CLASSES)
    ])


def simple_model_activation_hyperbolic_tangent():
    return tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(32, (3, 3),
                               activation='tanh',
                               input_shape=(IMG_HEIGHT, IMG_WIDTH, RGB_DEPTH)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CLASSES)
    ])


# 4 custom models with different activation functions
def model_custom_activation_relu():
    return tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(16, 3,
                               input_shape=(IMG_HEIGHT, IMG_WIDTH, RGB_DEPTH),
                               activation='relu',
                               padding='same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3,
                               input_shape=(IMG_HEIGHT, IMG_WIDTH, RGB_DEPTH),
                               activation='relu',
                               padding='same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3,
                               input_shape=(IMG_HEIGHT, IMG_WIDTH, RGB_DEPTH),
                               activation='relu',
                               padding='same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3,
                               input_shape=(IMG_HEIGHT, IMG_WIDTH, RGB_DEPTH),
                               activation='relu',
                               padding='same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(NUM_CLASSES)
    ])


def model_custom_activation_leaky_relu():
    activation_alpha = 0.01
    return tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(
            16,
            3,
            input_shape=(IMG_HEIGHT, IMG_WIDTH, RGB_DEPTH),
            activation=partial(tf.nn.leaky_relu, alpha=activation_alpha),
            padding='same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(
            32,
            3,
            input_shape=(IMG_HEIGHT, IMG_WIDTH, RGB_DEPTH),
            activation=partial(tf.nn.leaky_relu, alpha=activation_alpha),
            padding='same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(
            32,
            3,
            input_shape=(IMG_HEIGHT, IMG_WIDTH, RGB_DEPTH),
            activation=partial(tf.nn.leaky_relu, alpha=activation_alpha),
            padding='same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(
            32,
            3,
            input_shape=(IMG_HEIGHT, IMG_WIDTH, RGB_DEPTH),
            activation=partial(tf.nn.leaky_relu, alpha=activation_alpha),
            padding='same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(NUM_CLASSES)
    ])


def model_custom_activation_sigmoid():
    return tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(
            16,
            3,
            input_shape=(IMG_HEIGHT, IMG_WIDTH, RGB_DEPTH),
            activation='sigmoid',
            padding='same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(
            32,
            3,
            input_shape=(IMG_HEIGHT, IMG_WIDTH, RGB_DEPTH),
            activation='sigmoid',
            padding='same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(
            32,
            3,
            input_shape=(IMG_HEIGHT, IMG_WIDTH, RGB_DEPTH),
            activation='sigmoid',
            padding='same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(
            32,
            3,
            input_shape=(IMG_HEIGHT, IMG_WIDTH, RGB_DEPTH),
            activation='sigmoid',
            padding='same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(NUM_CLASSES)
    ])


def model_custom_activation_hyperbolic_tangent():
    return tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(
            16,
            3,
            input_shape=(IMG_HEIGHT, IMG_WIDTH, RGB_DEPTH),
            activation='tanh',
            padding='same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(
            32,
            3,
            input_shape=(IMG_HEIGHT, IMG_WIDTH, RGB_DEPTH),
            activation='tanh',
            padding='same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(
            32,
            3,
            input_shape=(IMG_HEIGHT, IMG_WIDTH, RGB_DEPTH),
            activation='tanh',
            padding='same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(
            32,
            3,
            input_shape=(IMG_HEIGHT, IMG_WIDTH, RGB_DEPTH),
            activation='tanh',
            padding='same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(NUM_CLASSES)
    ])


# Some already defined models
def model_vgg16():
    base_model = tf.keras.applications.VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, RGB_DEPTH))
    return tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(NUM_CLASSES)
    ])


def model_inception_v3():
    base_model = tf.keras.applications.InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, RGB_DEPTH))
    return tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(NUM_CLASSES)
    ])


def model_resnet50():
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, RGB_DEPTH))
    return tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(NUM_CLASSES)
    ])


# Compile chosen simple model
def create_simple_model(activation_type: str):
    if activation_type == 'leaky_relu':
        my_model = simple_model_activation_leaky_relu()
    elif activation_type == 'sigmoid':
        my_model = simple_model_activation_sigmoid()
    elif activation_type == 'hyperbolic_tangent':
        my_model = simple_model_activation_hyperbolic_tangent()
    else:
        my_model = simple_model_activation_relu()

    my_model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return my_model


# Compile chosen custom model
def create_custom_model(activation_type: str):
    if activation_type == 'leaky_relu':
        my_model = model_custom_activation_leaky_relu()
    elif activation_type == 'sigmoid':
        my_model = model_custom_activation_sigmoid()
    elif activation_type == 'hyperbolic_tangent':
        my_model = model_custom_activation_hyperbolic_tangent()
    else:
        my_model = model_custom_activation_relu()

    my_model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return my_model


def create_model_vgg16():
    my_model = model_vgg16()
    my_model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return my_model


def create_model_inception_v3():
    my_model = model_inception_v3()
    my_model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return my_model


def create_model_resnet50():
    my_model = model_resnet50()
    my_model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return my_model


def choose_activation_type():
    print('Choose activation type:')
    print('1. ReLU')
    print('2. Leaky ReLU')
    print('3. Sigmoid')
    print('4. Hyperbolic Tangent')
    activation_type = int(input('Enter number: '))
    if activation_type == 1:
        return 'relu'
    elif activation_type == 2:
        return 'leaky_relu'
    elif activation_type == 3:
        return 'sigmoid'
    elif activation_type == 4:
        return 'hyperbolic_tangent'
    else:
        print('Wrong choice, defaulting to ReLU')
        return 'relu'


def choose_model():
    print('Choose a model:')
    print('1. Simple model')
    print('2. Custom model')
    print('3. VGG16')
    print('4. InceptionV3')
    print('5. ResNet50')
    print('6. Exit')

    user_choice = int(input("Choose model: "))
    if user_choice == 1:
        print("Simple model")
        activation_type = choose_activation_type()
        return create_simple_model(activation_type)
    elif user_choice == 2:
        print("Custom model")
        activation_type = choose_activation_type()
        return create_custom_model(activation_type)
    elif user_choice == 3:
        print("VGG16 model")
        change_batch_size(VGG16_MODEL_BATCH_SIZE)
        change_epoch(VGG16_INCEPTION_V3_RESNET50_EPOCHS)
        return create_model_vgg16()
    elif user_choice == 4:
        print("Inception V3 model")
        change_batch_size(INCEPTION_V3_MODEL_BATCH_SIZE)
        change_epoch(VGG16_INCEPTION_V3_RESNET50_EPOCHS)
        return create_model_inception_v3()
    elif user_choice == 5:
        print("ResNet50 model")
        change_batch_size(RESNET50_MODEL_BATCH_SIZE)
        change_epoch(VGG16_INCEPTION_V3_RESNET50_EPOCHS)
        return create_model_resnet50()
    elif user_choice == 6:
        print("Exit")
        exit()
    else:
        print("Invalid choice, using default model!")
        activation_type = choose_activation_type()
        return create_simple_model(activation_type)


def run_all_models_and_save_results():
    activation_types = ['relu', 'leaky_relu', 'sigmoid', 'hyperbolic_tangent']
    print("Running all models and saving results...")

    print("Simple models...")
    for activation_type in activation_types:
        print("Activation type: " + activation_type)
        simple_model = create_simple_model(activation_type)
        train_model(simple_model)
        simple_model.save('./saved_models/simple_model_activation_{0}_epochs_{1}.h5'.format(activation_type, EPOCHS))
        with open('./results/simple_model_activation_{0}_epochs_{1}.txt'.format(activation_type, EPOCHS), 'w') as f:
            f.write("Activation type: {0}\n".format(activation_type))
            f.write("\t-> Accuracy: {0}\n".format(str(simple_model.history.history['accuracy'][EPOCHS - 1])))
            f.write("\t-> Loss: {0}\n".format(str(simple_model.history.history['loss'][EPOCHS - 1])))
            f.write("\t-> Validation accuracy: {0}\n".format(str(simple_model.history.history['val_accuracy'][EPOCHS - 1])))
            f.write("\t-> Validation loss: {0}\n".format(str(simple_model.history.history['val_loss'][EPOCHS - 1])))
        del simple_model

    print("Custom models...")
    for activation_type in activation_types:
        print("Activation type: " + activation_type)
        custom_model = create_custom_model(activation_type)
        train_model(custom_model)
        custom_model.save('./saved_models/custom_model_activation_{0}_epochs_{1}.h5'.format(activation_type, EPOCHS))
        with open('./results/custom_model_activation_{0}_epochs_{1}.txt'.format(activation_type, EPOCHS), 'w') as f:
            f.write("Activation type: {0}\n".format(activation_type))
            f.write("\t-> Accuracy: {0}\n".format(str(custom_model.history.history['accuracy'][EPOCHS - 1])))
            f.write("\t-> Loss: {0}\n".format(str(custom_model.history.history['loss'][EPOCHS - 1])))
            f.write("\t-> Validation accuracy: {0}\n".format(str(custom_model.history.history['val_accuracy'][EPOCHS - 1])))
            f.write("\t-> Validation loss: {0}\n".format(str(custom_model.history.history['val_loss'][EPOCHS - 1])))
        del custom_model

    try:
        print("VGG16 model")
        change_batch_size(VGG16_MODEL_BATCH_SIZE)
        change_epoch(VGG16_INCEPTION_V3_RESNET50_EPOCHS)
        vgg16 = create_model_vgg16()
        train_model(vgg16)
        vgg16.save('./saved_models/vgg16_model_epochs_{0}.h5'.format(EPOCHS))
        with open('./results/vgg16.txt', 'w') as f:
            f.write("\t-> Accuracy: {0}\n".format(str(vgg16.history.history['accuracy'][EPOCHS - 1])))
            f.write("\t-> Loss: {0}\n".format(str(vgg16.history.history['loss'][EPOCHS - 1])))
            f.write("\t-> Validation accuracy: {0}\n".format(str(vgg16.history.history['val_accuracy'][EPOCHS - 1])))
            f.write("\t-> Validation loss: {0}\n".format(str(vgg16.history.history['val_loss'][EPOCHS - 1])))
        del vgg16
    except Exception:
        print("VGG16 model failed!")

    try:
        print("Inception V3 model")
        change_batch_size(INCEPTION_V3_MODEL_BATCH_SIZE)
        change_epoch(VGG16_INCEPTION_V3_RESNET50_EPOCHS)
        inception_v3 = create_model_inception_v3()
        train_model(inception_v3)
        inception_v3.save('./saved_models/inception_v3_model_epochs_{0}.h5'.format(EPOCHS))
        with open('./results/inception_v3.txt', 'w') as f:
            f.write("\t-> Accuracy: {0}\n".format(str(inception_v3.history.history['accuracy'][EPOCHS - 1])))
            f.write("\t-> Loss: {0}\n".format(str(inception_v3.history.history['loss'][EPOCHS - 1])))
            f.write("\t-> Validation accuracy: {0}\n".format(str(inception_v3.history.history['val_accuracy'][EPOCHS - 1])))
            f.write("\t-> Validation loss: {0}\n".format(str(inception_v3.history.history['val_loss'][EPOCHS - 1])))
        del inception_v3
    except Exception:
        print("Inception V3 model failed!")

    try:
        print("ResNet50 model")
        change_batch_size(RESNET50_MODEL_BATCH_SIZE)
        change_epoch(VGG16_INCEPTION_V3_RESNET50_EPOCHS)
        resnet50 = create_model_resnet50()
        train_model(resnet50)
        resnet50.save('./saved_models/resnet50_model_epochs_{0}.h5'.format(EPOCHS))
        with open('./results/resnet50.txt', 'w') as f:
            f.write("\t-> Accuracy: {0}\n".format(str(resnet50.history.history['accuracy'][EPOCHS - 1])))
            f.write("\t-> Loss: {0}\n".format(str(resnet50.history.history['loss'][EPOCHS - 1])))
            f.write("\t-> Validation accuracy: {0}\n".format(str(resnet50.history.history['val_accuracy'][EPOCHS - 1])))
            f.write("\t-> Validation loss: {0}\n".format(str(resnet50.history.history['val_loss'][EPOCHS - 1])))
        del resnet50
    except Exception:
        print("ResNet50 model failed!")


# model.build(input_shape=(None, img_height, img_width, 3))
# print(model.summary())


def train_model(my_model):
    TRAIN_DATASET = init_train_dataset()
    VALIDATION_DATASET = init_validation_dataset()
    TEST_DATASET = init_test_dataset()

    my_model.fit(
        TRAIN_DATASET,
        validation_data=VALIDATION_DATASET,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )


if __name__ == '__main__':
    run_all_models_and_save_results()
